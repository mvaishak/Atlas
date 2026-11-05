"""Observability must preserve the real fixture prediction boundary."""

import math
from pathlib import Path

from fastapi.testclient import TestClient
from test_m7_api import setup
from test_ranking_parity import configured

from atlasrank.evaluation.ranking import offline_predict
from atlasrank.ranking.prediction import assert_exact_parity
from atlasrank.serving.app import create_app
from atlasrank.serving.metrics import Metrics


def test_timed_prediction_preserves_features_and_order(tmp_path: Path) -> None:
    context, ranking, observations, _ = configured(tmp_path)
    candidates = [("us", "c"), ("us", "b"), ("us", "a")]
    query = " ＲＥＤ\tapple apple "
    expected = offline_predict(context, ranking.ranker, query, candidates)
    ranked, timings = ranking.predict_with_timings(query, candidates)
    assert ranked == expected == [(("us", "a"), 0.5), (("us", "b"), 0.5), (("us", "c"), 0.5)]
    assert set(timings) == {"feature_construction", "ranker_prediction", "ranking_order"}
    assert all(math.isfinite(value) and value >= 0 for value in timings.values())
    assert ranking.predict(query, candidates) == expected
    assert_exact_parity(observations[0], observations[1])
    assert_exact_parity(observations[0], observations[2])


def test_search_reports_dispatch_and_nested_ranking_stages(tmp_path: Path) -> None:
    _, runtime, _ = setup(tmp_path)
    with TestClient(create_app(serving_runtime=runtime)) as client:
        response = client.post("/search", json={"query": "apple", "pipeline": "m6_xgb_m4"})
        assert response.status_code == 200
        timings = response.json()["timing_ms"]
        assert set(timings) == {
            "lexical",
            "encoding",
            "dense_retrieval",
            "fusion",
            "features_and_prediction",
            "feature_construction",
            "ranker_prediction",
            "ranking_order",
            "total",
            "request_dispatch",
        }
        assert all(math.isfinite(value) and value >= 0 for value in timings.values())
        assert (
            sum(
                timings[name]
                for name in ("feature_construction", "ranker_prediction", "ranking_order")
            )
            <= timings["features_and_prediction"]
            <= timings["total"]
        )
        metrics = client.get("/metrics").text
        for name in (
            "request_dispatch",
            "feature_construction",
            "ranker_prediction",
            "ranking_order",
        ):
            assert f'pipeline="m6_xgb_m4",stage="{name}"' in metrics


def test_new_stage_metrics_keep_labels_bounded() -> None:
    metrics = Metrics()
    values = {
        name: 2.0
        for name in (
            "request_dispatch",
            "feature_construction",
            "ranker_prediction",
            "ranking_order",
            "unbounded",
        )
    }
    metrics.stages("m6_xgb_m4", values)
    metrics.stages("unknown-pipeline", values)
    assert metrics.stage_count == {("m6_xgb_m4", name): 1 for name in values if name != "unbounded"}
    assert all(value == 0.002 for value in metrics.stage_sum.values())
