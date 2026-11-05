"""Benchmark-only encoding observation; nested timings are not additive wall time."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlasrank.artifacts.serving_bundle import Pipeline
from atlasrank.retrieval.chunks import Encoder
from atlasrank.serving.runtime import PipelineRuntime


class TimedEncoder:
    """Forward the original recipe and vectors without changing encoder behavior."""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.recipe = encoder.recipe
        self.calls_ms: list[float] = []

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if len(self.calls_ms) >= 8:
            raise RuntimeError("Unexpected number of encoding calls in profiling request")
        started = time.perf_counter()
        result = self.encoder.encode(texts)
        self.calls_ms.append((time.perf_counter() - started) * 1000)
        return result


class ProfiledRuntime(PipelineRuntime):
    """Wrap a dedicated benchmark runtime, sharing its context and admission lock."""

    def __init__(self, runtime: PipelineRuntime) -> None:
        if runtime.context is not None and isinstance(runtime.context.encoder, TimedEncoder):
            raise ValueError("Runtime context is already profiled")
        super().__init__(
            runtime.lexical,
            runtime.bundle_id,
            runtime.pipelines,
            runtime.context,
            runtime.ranking,
            runtime.maximum_rss_bytes,
        )
        self.admission = runtime.admission
        self.timed_encoder = TimedEncoder(self.context.encoder) if self.context else None
        if self.context is not None and self.timed_encoder is not None:
            self.context.encoder = self.timed_encoder

    def _search(self, query: str, pipeline: Pipeline, top_k: int) -> dict[str, Any]:
        if self.timed_encoder is not None:
            self.timed_encoder.calls_ms.clear()
        result = super()._search(query, pipeline, top_k)
        calls = self.timed_encoder.calls_ms if self.timed_encoder is not None else []
        result["timing_ms"]["query_encoding_total"] = sum(calls)
        result["timing_ms"]["feature_query_encoding"] = (
            sum(calls[1:]) if pipeline == "m6_xgb_m4" else 0.0
        )
        return result
