"""Live pipeline contracts use the real HTTP and offline prediction boundaries."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_ranking_parity import configured

from atlasrank.evaluation.ranking import offline_predict
from atlasrank.ranking.prediction import assert_exact_parity
from atlasrank.serving.app import create_app


def setup(tmp_path: Path) -> tuple[Any, Any, list[Any]]:
    from atlasrank.serving.runtime import PipelineRuntime

    context, ranking, observations, _ = configured(tmp_path)
    return (
        context,
        PipelineRuntime(
            context.lexical,
            "fixture-serving",
            ("bm25", "dense_m4", "rrf_m4", "m6_xgb_m4"),
            context,
            ranking,
        ),
        observations,
    )


def test_live_pipeline_exact_prediction_boundary(tmp_path: Path) -> None:
    context, runtime, observations = setup(tmp_path)
    # Independently retrieve the offline reference pool, then compare actual predictor input.
    from atlasrank.retrieval.fusion import reciprocal_rank_fusion

    query = " ＲＥＤ apple "
    lexical = [(h.product_locale, h.product_id) for h in context.lexical.search(query, 100)]
    dense = [k for k, _ in context.dense.search(context.encoder.encode([query]), 100)]
    pool = [k for k, _ in reciprocal_rank_fusion([lexical, dense], 100)]
    expected = offline_predict(context, runtime.ranking.ranker, query, pool)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post("/search", json={"query": query, "pipeline": "m6_xgb_m4"})
        assert response.status_code == 200
        data = response.json()
        assert data["experimental"] and data["execution"] == "deterministic_fixture"
        assert data["candidate_count"] == len(pool)
        assert [(("us", r["product_id"]), r["score"]) for r in data["results"]] == expected
        assert all(v >= 0 for v in data["timing_ms"].values())
        assert data["request_id"] == response.headers["x-request-id"]
    assert_exact_parity(observations[0], observations[1])


def test_selection_empty_results_errors_and_legacy_limits(tmp_path: Path) -> None:
    _, runtime, _ = setup(tmp_path)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        for p in runtime.pipelines:
            response = client.post("/search", json={"query": "apple", "pipeline": p, "top_k": 100})
            assert response.status_code == 200
            assert (
                len({r["product_id"] for r in response.json()["results"]})
                == response.json()["returned"]
            )
        assert (
            client.post("/search", json={"query": "no_matching_term", "pipeline": "bm25"}).json()[
                "results"
            ]
            == []
        )
        for body in (
            {"query": "apple", "pipeline": "other"},
            {"query": "apple", "top_k": True},
            {"query": "apple", "locale": "jp"},
            {"query": " "},
            {"query": "x" * 513},
        ):
            assert client.post("/search", json=body).status_code == 422
        assert client.post("/search", content="{bad").status_code == 422


def test_unavailable_complete_bundle_fails_closed(tmp_path: Path) -> None:
    with TestClient(
        create_app(tmp_path / "missing", serving_bundle=tmp_path / "bad", artifact_root=tmp_path)
    ) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/search", json={"query": "apple"}).status_code == 503


def test_admission_is_bounded_and_metrics_logs_are_private(tmp_path: Path, caplog: Any) -> None:
    _, runtime, _ = setup(tmp_path)
    import logging

    caplog.set_level(logging.INFO, logger="atlasrank.http")
    with TestClient(create_app(serving_runtime=runtime)) as client:
        with runtime.admission:
            response = client.post("/search", json={"query": "private_query", "pipeline": "rrf_m4"})
            assert (
                response.status_code == 503 and response.json()["error"]["code"] == "pipeline_busy"
            )
        assert client.post("/search", content="x" * 16385).status_code == 413
        client.post("/search", json={"query": "private_query", "pipeline": "bm25"})
        client.get("/arbitrary-user-path")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "atlasrank_requests_total" in response.text
        assert "private_query" not in response.text and "arbitrary-user-path" not in response.text
    assert "private_query" not in caplog.text
    for record in caplog.records:
        if record.name == "atlasrank.http":
            assert isinstance(json.loads(record.message), dict)


def test_disabled_pipeline_never_falls_back(tmp_path: Path) -> None:
    _, runtime, _ = setup(tmp_path)
    runtime.pipelines = ("bm25",)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        response = client.post("/search", json={"query": "apple", "pipeline": "rrf_m4"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "pipeline_unavailable"


def test_internal_key_error_is_sanitized(tmp_path: Path, monkeypatch: Any) -> None:
    _, runtime, _ = setup(tmp_path)

    def broken(*args: Any) -> Any:
        raise KeyError("private-product-or-query")

    monkeypatch.setattr(runtime.lexical, "search", broken)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        response = client.post("/search", json={"query": "apple"})
        assert response.status_code == 500
        assert "private-product-or-query" not in response.text


def test_chunked_body_limit_counts_one_request(tmp_path: Path) -> None:
    _, runtime, _ = setup(tmp_path)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        response = client.post("/search", content=iter([b"x" * 9000, b"y" * 9000]))
        assert response.status_code == 413
        assert 'route="/search",status="413"} 1' in client.get("/metrics").text


def test_feature_fault_in_live_http_is_detected(tmp_path: Path, monkeypatch: Any) -> None:
    from dataclasses import replace

    import atlasrank.serving.ranking as http

    _, runtime, observations = setup(tmp_path)
    runtime.search("apple", "m6_xgb_m4", 10)
    original = http.build_features

    def broken(*args: Any, **kwargs: Any) -> Any:
        batch = original(*args, **kwargs)
        values = batch.values.copy()
        values[:, [0, 1]] = values[:, [1, 0]]
        return replace(batch, values=values)

    monkeypatch.setattr(http, "build_features", broken)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        assert (
            client.post("/search", json={"query": "apple", "pipeline": "m6_xgb_m4"}).status_code
            == 200
        )
    with pytest.raises(ValueError, match="parity"):
        assert_exact_parity(observations[0], observations[1])


def test_resource_failure_removes_readiness(tmp_path: Path, monkeypatch: Any) -> None:
    _, runtime, _ = setup(tmp_path)

    def exhausted(*args: Any) -> Any:
        raise MemoryError("private-details")

    monkeypatch.setattr(runtime, "search", exhausted)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        assert client.post("/search", json={"query": "apple"}).status_code == 503
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 200


def test_complete_rerank_shares_search_admission(tmp_path: Path) -> None:
    _, runtime, _ = setup(tmp_path)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        with runtime.admission:
            response = client.post("/rerank", json={"query": "apple", "product_ids": ["a"]})
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "pipeline_busy"


def test_dependency_load_failure_keeps_liveness(tmp_path: Path, monkeypatch: Any) -> None:
    import atlasrank.serving.app as http

    def missing(*args: Any) -> Any:
        raise ImportError("private-environment-path")

    monkeypatch.setattr(http, "load_runtime", missing)
    with TestClient(create_app(serving_bundle=tmp_path, artifact_root=tmp_path)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_body_rejection_uses_same_request_id(caplog: Any) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="atlasrank.http")
    with TestClient(create_app()) as client:
        response = client.post("/search", content=b"x" * 16385)
    rejected = [
        json.loads(r.message)
        for r in caplog.records
        if r.name == "atlasrank.http" and json.loads(r.message)["event"] == "request_rejected"
    ]
    assert rejected[0]["request_id"] == response.headers["x-request-id"]
