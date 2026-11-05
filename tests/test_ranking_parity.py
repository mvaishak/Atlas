"""Independent offline/HTTP flows captured inside the actual fixture predictor."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_ranking_features import runtime

from atlasrank.serving.app import create_app


def configured(root: Path) -> tuple[Any, Any, list[Any], Path]:
    from atlasrank.ranking.prediction import DeterministicRanker, save_deterministic_fixture
    from atlasrank.serving.ranking import RankingRuntime

    context = runtime(root)
    bundle = root / "fixture-ranker"
    save_deterministic_fixture(bundle, context.provenance)
    observations: list[Any] = []
    ranker = DeterministicRanker.load(bundle, context.provenance)
    actual_model = ranker.model

    class RecordingModel:
        trained = False

        def predict(self, inputs: Any) -> Any:
            observations.append(inputs)  # Capture inside the actual prediction method.
            return actual_model.predict(inputs)

    ranker.model = RecordingModel()
    return context, RankingRuntime(context, ranker), observations, bundle


def test_actual_predictor_input_parity_with_permuted_candidates_and_round_trip(
    tmp_path: Path,
) -> None:
    from atlasrank.evaluation.ranking import offline_predict
    from atlasrank.ranking.prediction import assert_exact_parity

    context, ranking, observed, bundle = configured(tmp_path)
    query = " ＲＥＤ\tapple apple "
    offline = offline_predict(
        context, ranking.ranker, query, [("us", "c"), ("us", "b"), ("us", "a")]
    )
    assert len(observed) == 1
    with TestClient(create_app(tmp_path / "lexical", ranking=ranking)) as client:
        r = client.post("/rerank", json={"query": query, "product_ids": ["b", "a", "c"]})
        assert r.status_code == 200
        assert r.json()["mode"] == "fixed_candidate_fixture"
        assert r.json()["learned"] is False
        assert [x["product_id"] for x in r.json()["results"]] == ["a", "b", "c"]
        assert [x["score"] for x in r.json()["results"]] == [0.5, 0.5, 0.5]
        assert client.get("/health/ready").json()["ready"] is True
        assert client.post("/search", json={"query": "apple"}).status_code == 200
    assert len(observed) == 2
    assert_exact_parity(observed[0], observed[1])
    assert np.array_equal(
        observed[0].values[:, 1:],
        np.array([[1, 1, 3, 3], [0, 0, 3, 0], [0.6, 0, 3, 2]], dtype=np.float32),
    )
    assert observed[0].values.dtype == np.float32 and not observed[0].missing.any()
    assert observed[0].feature_types == ("float32",) * 5
    assert observed[0].keys == (("us", "a"), ("us", "b"), ("us", "c"))
    assert offline == [(("us", "a"), 0.5), (("us", "b"), 0.5), (("us", "c"), 0.5)]
    from atlasrank.ranking.prediction import DeterministicRanker

    again = DeterministicRanker.load(bundle, context.provenance)
    assert again.model.trained is False


@pytest.mark.parametrize("fault", ["swapped_values", "changed_normalization"])
def test_final_boundary_detects_serving_only_feature_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import atlasrank.serving.ranking as http
    from atlasrank.evaluation.ranking import offline_predict
    from atlasrank.ranking.prediction import assert_exact_parity

    context, ranking, observed, _ = configured(tmp_path)
    query = " ＲＥＤ apple apple "
    offline_predict(context, ranking.ranker, query, [("us", "a"), ("us", "c")])
    original = http.build_features

    def broken(*args: Any, **kwargs: Any) -> Any:
        batch = original(*args, **kwargs)
        values = batch.values.copy()
        if fault == "swapped_values":
            values[:, [0, 1]] = values[:, [1, 0]]
        else:
            # Simulate an HTTP-only raw-ASCII token path that loses the fullwidth term.
            values[:, 2] = [0.5, 0]
            values[:, 3] = 2
        return replace(batch, values=values)

    monkeypatch.setattr(http, "build_features", broken)
    with TestClient(create_app(tmp_path / "lexical", ranking=ranking)) as client:
        assert (
            client.post("/rerank", json={"query": query, "product_ids": ["c", "a"]}).status_code
            == 200
        )
    with pytest.raises(ValueError, match="parity"):
        assert_exact_parity(observed[0], observed[1])


@pytest.mark.parametrize("fault", ["names", "identities", "dtype", "missing"])
def test_invalid_serving_batch_fails_before_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import atlasrank.serving.ranking as http

    _, ranking, observed, _ = configured(tmp_path)
    original = http.build_features

    def broken(*args: Any, **kwargs: Any) -> Any:
        b = original(*args, **kwargs)
        if fault == "names":
            return replace(b, names=tuple(reversed(b.names)))
        if fault == "identities":
            return replace(b, keys=tuple(reversed(b.keys)))
        if fault == "dtype":
            return replace(b, values=b.values.astype(np.float64))
        masks = b.missing.copy()
        masks[0, 0] = True
        return replace(b, missing=masks)

    monkeypatch.setattr(http, "build_features", broken)
    with TestClient(create_app(tmp_path / "lexical", ranking=ranking)) as client:
        r = client.post("/rerank", json={"query": "apple", "product_ids": ["b", "a"]})
        assert r.status_code == 500 and r.json()["error"]["code"] == "feature_contract_failed"
    assert not observed


def test_unavailable_and_bad_candidates_preserve_old_api(tmp_path: Path) -> None:
    context, ranking, observed, _ = configured(tmp_path)
    with TestClient(create_app(tmp_path / "lexical")) as client:
        assert (
            client.post("/rerank", json={"query": "apple", "product_ids": ["a"]}).status_code == 503
        )
        assert client.get("/health/ready").json() == {
            "ready": True,
            "bundle_id": context.lexical.bundle_id,
        }
    with TestClient(create_app(tmp_path / "lexical", ranking=ranking)) as client:
        for ids in [[], ["a", "a"], ["missing"]]:
            assert (
                client.post("/rerank", json={"query": "apple", "product_ids": ids}).status_code
                == 422
            )
    assert not observed


@pytest.mark.parametrize("fault", ["missing", "nan", "duplicate"])
def test_http_required_score_failure_never_reaches_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    context, ranking, observed, _ = configured(tmp_path)

    def bad_scores(*_: Any) -> Any:
        if fault == "missing":
            return []
        if fault == "nan":
            return [(("us", "a"), float("nan")), (("us", "b"), 0.0)]
        return [(("us", "a"), 0.0), (("us", "a"), 0.0)]

    monkeypatch.setattr(context.dense, "candidate_scores", bad_scores)
    with TestClient(create_app(tmp_path / "lexical", ranking=ranking)) as client:
        r = client.post("/rerank", json={"query": "apple", "product_ids": ["a", "b"]})
        assert r.status_code == 500
        assert r.json() == {"error": {"code": "feature_contract_failed"}}
    assert not observed


@pytest.mark.parametrize("field", ["model_id", "query_recipe_id", "dense_bundle_id"])
def test_fixture_bundle_rejects_other_model_or_encoder(tmp_path: Path, field: str) -> None:
    from atlasrank.ranking.prediction import DeterministicRanker

    context, ranking, observed, bundle = configured(tmp_path)
    incompatible = replace(context.provenance, **{field: "wrong"})
    with pytest.raises(ValueError, match="bundle"):
        DeterministicRanker.load(bundle, incompatible)
    ranking.ranker.provenance = incompatible
    with TestClient(create_app(tmp_path / "lexical", ranking=ranking)) as client:
        assert (
            client.post("/rerank", json={"query": "apple", "product_ids": ["a"]}).status_code == 503
        )
        assert client.get("/health/ready").json()["ready"] is True
        assert client.post("/search", json={"query": "apple"}).status_code == 200
    assert not observed


def test_corrupt_model_fixture_rejected_without_overwrite(tmp_path: Path) -> None:
    from atlasrank.ranking.prediction import DeterministicRanker, save_deterministic_fixture

    context, _, _, bundle = configured(tmp_path)
    with pytest.raises(FileExistsError):
        save_deterministic_fixture(bundle, context.provenance)
    model = bundle / "model.json"
    model.chmod(0o644)
    model.write_text("{}")
    with pytest.raises(ValueError):
        DeterministicRanker.load(bundle, context.provenance)


def test_one_ulp_difference_fails_exact_parity(tmp_path: Path) -> None:
    from atlasrank.evaluation.ranking import offline_predict
    from atlasrank.ranking.prediction import assert_exact_parity

    context, ranking, observed, _ = configured(tmp_path)
    offline_predict(context, ranking.ranker, "apple", [("us", "a")])
    values = observed[0].values.copy()
    values[0, 1] = np.nextafter(values[0, 1], np.float32(0))
    assert np.allclose(values, observed[0].values)  # Approximate equality is insufficient.
    with pytest.raises(ValueError, match="parity"):
        assert_exact_parity(observed[0], replace(observed[0], values=values))


def test_batching_candidates_and_empty_text_preserve_final_rows(tmp_path: Path) -> None:
    from atlasrank.evaluation.ranking import offline_predict

    context, ranking, observed, _ = configured(tmp_path)
    offline_predict(
        context, ranking.ranker, " ＲＥＤ apple apple ", [("us", "d"), ("us", "b"), ("us", "a")]
    )
    reference = observed[0]
    with TestClient(create_app(tmp_path / "lexical", ranking=ranking)) as client:
        for ids in [["a"], ["b", "d"]]:
            assert (
                client.post(
                    "/rerank", json={"query": "red apple apple", "product_ids": ids}
                ).status_code
                == 200
            )
    actual = {
        key: (b.values[i], b.missing[i], b.text_empty[i])
        for b in observed[1:]
        for i, key in enumerate(b.keys)
    }
    for i, key in enumerate(reference.keys):
        for column, expected in enumerate(
            [reference.values[i], reference.missing[i], reference.text_empty[i]]
        ):
            assert np.array_equal(actual[key][column], expected)


@pytest.mark.parametrize(
    "body",
    [
        {"query": " "},
        {"query": "apple", "product_ids": [True]},
        {"query": "apple", "product_ids": ["a"], "locale": "jp"},
        {"query": "apple", "product_ids": ["a"], "extra": 1},
    ],
)
def test_rerank_uses_existing_query_validation(body: dict[str, Any]) -> None:
    with TestClient(create_app()) as client:
        assert client.post("/rerank", json=body).status_code == 422
