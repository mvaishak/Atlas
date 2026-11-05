"""Benchmark-only encoder timing preserves exact inputs and outputs."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from test_m7_api import setup

from atlasrank.artifacts.dense import digest
from atlasrank.artifacts.serving_bundle import Pipeline
from atlasrank.ranking.prediction import assert_exact_parity


def test_profiled_pipeline_preserves_features_outputs_and_recipe(tmp_path: Path) -> None:
    from atlasrank.benchmarking.profiling import ProfiledRuntime

    context, runtime, observations = setup(tmp_path)
    query = " ＲＥＤ\tapple apple "
    expected = runtime.search(query, "m6_xgb_m4", 10)
    recipe = context.encoder.recipe
    recipe_id = digest(recipe)
    profiled = ProfiledRuntime(runtime)
    actual = profiled.search(query, "m6_xgb_m4", 10)
    assert actual["results"] == expected["results"]
    assert_exact_parity(observations[0], observations[1])
    assert context.encoder.recipe is recipe
    assert digest(context.encoder.recipe) == recipe_id == context.provenance.query_recipe_id
    assert profiled.admission is runtime.admission
    assert profiled.context is runtime.context and profiled.ranking is runtime.ranking
    assert profiled.timed_encoder is not None
    assert len(profiled.timed_encoder.calls_ms) == 2
    timings = actual["timing_ms"]
    assert timings["query_encoding_total"] == sum(profiled.timed_encoder.calls_ms)
    assert timings["feature_query_encoding"] == profiled.timed_encoder.calls_ms[1]
    assert all(math.isfinite(value) and value >= 0 for value in timings.values())
    pipelines: list[tuple[Pipeline, int]] = [("rrf_m4", 1), ("bm25", 0), ("m6_xgb_m4", 2)]
    for pipeline, call_count in pipelines:
        result = profiled.search(query, pipeline, 10)
        assert len(profiled.timed_encoder.calls_ms) == call_count
        if pipeline != "m6_xgb_m4":
            assert result["timing_ms"]["feature_query_encoding"] == 0
        if pipeline == "bm25":
            assert result["timing_ms"]["query_encoding_total"] == 0


def test_timed_encoder_preserves_array_identity_and_bounds_calls(tmp_path: Path) -> None:
    from atlasrank.benchmarking.profiling import TimedEncoder

    context, _, _ = setup(tmp_path)
    texts = ["apple"]
    expected: NDArray[np.float32] = context.encoder.encode(texts)

    class ExactEncoder:
        recipe: dict[str, Any] = context.encoder.recipe

        def encode(self, inputs: list[str]) -> NDArray[np.float32]:
            assert inputs is texts
            return expected

    timed = TimedEncoder(ExactEncoder())
    for _ in range(8):
        assert timed.encode(texts) is expected
    with pytest.raises(RuntimeError, match="encoding calls"):
        timed.encode(texts)
    assert len(timed.calls_ms) == 8


def test_profiled_runtime_rejects_nested_wrappers(tmp_path: Path) -> None:
    from atlasrank.benchmarking.profiling import ProfiledRuntime

    _, runtime, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="profiled"):
        ProfiledRuntime(ProfiledRuntime(runtime))
