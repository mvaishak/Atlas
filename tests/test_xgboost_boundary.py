"""Actual Linux native-boundary checks; never execute native prediction on Mac."""

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Native boundary is Linux-only")


def native_runtime(tmp_path: Path) -> tuple[Any, Any, Any]:
    from test_ranking_parity import configured

    from atlasrank.ranking.xgboost_boundary import ZeroUpdateModel

    context, runtime, _, _ = configured(tmp_path)
    predictor = ZeroUpdateModel()
    runtime.ranker.model = predictor
    return context, runtime, predictor


def test_real_native_offline_http_boundary_and_zero_update_round_trip(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from atlasrank.evaluation.ranking import offline_predict
    from atlasrank.ranking.prediction import assert_exact_parity
    from atlasrank.serving.app import create_app

    context, runtime, model = native_runtime(tmp_path)
    query = " ＲＥＤ\tapple apple "
    offline = offline_predict(
        context, runtime.ranker, query, [("us", "c"), ("us", "a"), ("us", "b")]
    )
    assert model.booster.num_boosted_rounds() == 0
    saved = tmp_path / "zero-update.ubj"
    model.booster.save_model(saved)
    model.booster = model.xgb.Booster(model_file=saved)
    model.booster.set_param({"nthread": 1, "device": "cpu"})
    with TestClient(create_app(tmp_path / "lexical", ranking=runtime)) as client:
        response = client.post("/rerank", json={"query": query, "product_ids": ["b", "c", "a"]})
        assert response.status_code == 200
        assert response.json()["learned"] is False
    assert len(model.observations) == 2
    before, after = model.observations
    assert_exact_parity(before, after)
    assert before.values.dtype == np.float32 and not before.missing.any()
    assert before.keys == (("us", "a"), ("us", "b"), ("us", "c"))
    assert (before.values == 0).any()
    assert np.array_equal(
        before.values[:, 1:],
        np.array([[1, 1, 3, 3], [0, 0, 3, 0], [0.6, 0, 3, 2]], dtype=np.float32),
    )
    assert [x["product_id"] for x in response.json()["results"]] == [k[1] for k, _ in offline]
    assert [x["score"] for x in response.json()["results"]] == [v for _, v in offline]
    assert all(v == 0.5 for _, v in offline)
    assert model.booster.num_boosted_rounds() == 0
    assert "torch" not in sys.modules
    destination = os.environ.get("ATLASRANK_XGB_EVIDENCE_DIR")
    if destination:
        lib = Path(model.xgb.core._LIB._name)
        report = {
            "kind": "zero_update_native_boundary_fixture",
            "boosted_rounds": 0,
            "names": list(before.names),
            "native_feature_types": ["float"] * 5,
            "dtype": str(before.values.dtype),
            "keys": before.keys,
            "values": before.values.tolist(),
            "missing": before.missing.tolist(),
            "text_empty": before.text_empty.tolist(),
            "provenance": asdict(before.provenance),
            "offline_http_exact": True,
            "save_load_exact": True,
            "predictions": [score for _, score in offline],
            "xgboost_version": model.xgb.__version__,
            "native_library_name": lib.name,
            "native_library_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
            "zero_update_model_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
        }
        (Path(destination) / "native-boundary.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize(
    "fault", ["columns", "names", "dtype", "missing_zero", "nan_substitution", "identities"]
)
def test_http_native_conversion_faults_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from fastapi.testclient import TestClient

    from atlasrank.evaluation.ranking import offline_predict
    from atlasrank.serving.app import create_app

    context, runtime, model = native_runtime(tmp_path)
    offline_predict(context, runtime.ranker, "apple", [("us", "a"), ("us", "b")])
    convert = model.convert

    def broken(inputs: Any) -> Any:
        values = inputs.values.copy()
        options: dict[str, Any] = {}
        if fault == "columns":
            values[:, [0, 1]] = values[:, [1, 0]]
        elif fault == "names":
            options["names"] = tuple(reversed(inputs.names))
        elif fault == "dtype":
            values = values.astype(np.float64)
        elif fault == "missing_zero":
            options["missing"] = 0.0
        elif fault == "nan_substitution":
            values[1, 0] = np.nan
        else:
            options["keys"] = tuple(reversed(inputs.keys))
        return convert(inputs, values=values, **options)

    monkeypatch.setattr(model, "convert", broken)
    with TestClient(create_app(tmp_path / "lexical", ranking=runtime)) as client:
        response = client.post("/rerank", json={"query": "apple", "product_ids": ["b", "a"]})
        assert response.status_code == 500
        assert response.json() == {"error": {"code": "feature_contract_failed"}}
    assert len(model.observations) == 1  # Invalid native inputs never reach Booster.predict.
    assert model.booster.num_boosted_rounds() == 0
