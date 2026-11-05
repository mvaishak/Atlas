"""One CPU runtime for frozen live pipelines; no fitting or judgment access."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import psutil
from numpy.typing import NDArray

from atlasrank.artifacts.dense import digest
from atlasrank.artifacts.query_recipe import validate_query_recipe
from atlasrank.artifacts.serving_bundle import Pipeline, component_path, preflight
from atlasrank.data.common import peak_rss_bytes
from atlasrank.evaluation.metrics import ProductKey
from atlasrank.features.ranking import DenseBinding, FeatureContext, model_identity
from atlasrank.retrieval.bm25 import LexicalIndex
from atlasrank.retrieval.fusion import reciprocal_rank_fusion
from atlasrank.serving.ranking import RankingRuntime


class DenseSearch(Protocol):
    def search(self, vector: NDArray[np.float32], k: int) -> list[tuple[ProductKey, float]]: ...


class PipelineUnavailable(Exception):
    pass


class PipelineBusy(Exception):
    pass


class PipelineRuntime:
    def __init__(
        self,
        lexical: LexicalIndex,
        bundle_id: str,
        pipelines: tuple[Pipeline, ...],
        context: FeatureContext | None = None,
        ranking: RankingRuntime | None = None,
        maximum_rss_bytes: int = 12 * 1024**3,
    ) -> None:
        self.lexical, self.bundle_id, self.pipelines = lexical, bundle_id, pipelines
        self.context, self.ranking = context, ranking
        self.maximum_rss_bytes = maximum_rss_bytes
        self.admission = threading.Lock()
        self.validate()

    def validate(self) -> None:
        if (
            not self.bundle_id
            or "bm25" not in self.pipelines
            or len(set(self.pipelines)) != len(self.pipelines)
        ):
            raise ValueError("Invalid serving identity or pipeline inventory")
        if any(p != "bm25" for p in self.pipelines):
            if self.context is None:
                raise ValueError("Neural pipelines require complete feature context")
            self.context.validate()
            if self.context.lexical.bundle_id != self.lexical.bundle_id:
                raise ValueError("Serving lexical context differs")
        if "m6_xgb_m4" in self.pipelines:
            if self.ranking is None or self.ranking.context is not self.context:
                raise ValueError("Matched ranking context required")
            self.ranking.validate(self.lexical)

    def search(self, query: str, pipeline: Pipeline, top_k: int) -> dict[str, Any]:
        if pipeline not in self.pipelines:
            raise PipelineUnavailable("pipeline_unavailable")
        with self.admitted():
            return self._search(query, pipeline, top_k)

    def rerank(self, query: str, candidates: list[ProductKey]) -> list[tuple[ProductKey, float]]:
        if self.ranking is None or "m6_xgb_m4" not in self.pipelines:
            raise PipelineUnavailable("pipeline_unavailable")
        with self.admitted():
            return self.ranking.predict(query, candidates)

    @contextmanager
    def admitted(self) -> Iterator[None]:
        if not self.admission.acquire(blocking=False):
            raise PipelineBusy("pipeline_busy")
        try:
            if peak_rss_bytes() > self.maximum_rss_bytes:
                raise MemoryError("Serving memory ceiling exceeded")
            yield
        finally:
            self.admission.release()

    def _search(self, query: str, pipeline: Pipeline, top_k: int) -> dict[str, Any]:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        stage = time.perf_counter()
        lexical = self.lexical.search(query, 100)
        timings["lexical"] = (time.perf_counter() - stage) * 1000
        ranked = [((h.product_locale, h.product_id), h.score) for h in lexical]
        dense_count = 0
        if pipeline != "bm25":
            if self.context is None:
                raise ValueError("Neural context unavailable")
            stage = time.perf_counter()
            vector = self.context.encoder.encode([query])
            timings["encoding"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            dense = cast(DenseSearch, self.context.dense).search(vector, 100)
            dense_count = len(dense)
            timings["dense_retrieval"] = (time.perf_counter() - stage) * 1000
            if pipeline == "dense_m4":
                ranked = dense
            else:
                stage = time.perf_counter()
                ranked = reciprocal_rank_fusion(
                    [[key for key, _ in ranked], [key for key, _ in dense]], 100
                )
                timings["fusion"] = (time.perf_counter() - stage) * 1000
                if pipeline == "m6_xgb_m4" and ranked:
                    if self.ranking is None:
                        raise ValueError("Ranker unavailable")
                    stage = time.perf_counter()
                    ranked, ranking_timings = self.ranking.predict_with_timings(
                        query, [key for key, _ in ranked]
                    )
                    timings.update(ranking_timings)
                    timings["features_and_prediction"] = (time.perf_counter() - stage) * 1000
        timings["total"] = (time.perf_counter() - started) * 1000
        return {
            "pipeline_id": pipeline,
            "bundle_id": self.bundle_id,
            "experimental": pipeline == "m6_xgb_m4",
            "execution": (
                "deterministic_fixture"
                if self.ranking is not None and not self.ranking.ranker.model.trained
                else "live"
            ),
            "score_type": {
                "bm25": "bm25",
                "dense_m4": "cosine",
                "rrf_m4": "rrf",
                "m6_xgb_m4": "ranker_margin",
            }[pipeline],
            "versions": {
                "lexical": self.lexical.bundle_id,
                "dense": self.context.provenance.dense_bundle_id if self.context else None,
                "ranker": self.ranking.ranker.artifact_id if self.ranking else None,
                "feature_schema": self.context.provenance.schema_id if self.context else None,
            },
            "candidate_count": len(ranked),
            "candidate_counts": {
                "lexical": len(lexical),
                "dense": dense_count,
                "final": len(ranked),
            },
            "results": [
                {
                    "product_locale": key[0],
                    "product_id": key[1],
                    "title": self.lexical.titles[self.lexical.row_for_key[key]],
                    "rank": i + 1,
                    "score": score,
                }
                for i, (key, score) in enumerate(ranked[:top_k])
            ],
            "returned": len(ranked[:top_k]),
            "timing_ms": timings,
        }


def load_runtime(
    directory: Path, root: Path, *, reference_query_recipe: dict[str, Any] | None = None
) -> PipelineRuntime:
    spec, identity = preflight(directory, root)
    # Must precede imports of either native FAISS or the neural encoder.
    if any(p != "bm25" for p in spec.pipelines) and sys.platform != "linux":
        raise ValueError("Combined neural/native serving requires compatible Linux CPU")
    if psutil.virtual_memory().available < spec.minimum_available_bytes:
        raise ValueError("Insufficient available memory for approved serving startup")
    from atlasrank.retrieval.bm25 import load

    paths = {k: component_path(root, v) for k, v in spec.components.items()}
    lexical = load(paths["lexical"])
    context, ranking = None, None
    if "dense" in paths:
        from atlasrank.retrieval.dense_bundle import load_bundle
        from atlasrank.retrieval.encoder import PinnedEncoder, configuration

        metadata = json.loads((paths["dense"] / "manifest.json").read_text())
        dense, products, metadata = load_bundle(paths["dense"], metadata["recipe"])
        if products["product_title"].to_list() != lexical.titles:
            raise ValueError("Dense/lexical catalog titles differ")
        encoder_spec = configuration(paths["dense_config"])
        encoder_spec["model_path"] = str(paths["model"])
        encoder = PinnedEncoder(encoder_spec)
        validate_query_recipe(encoder.recipe, str(spec.query_recipe_id), reference_query_recipe)
        for field in (
            "repo",
            "revision",
            "model_files_sha256",
            "dimensions",
            "max_length",
            "preprocessing",
            "padding",
            "pooling",
            "dtype",
            "attention",
            "encoder_code_sha256",
        ):
            if encoder.recipe[field] != metadata["recipe"][field]:
                raise ValueError("Frozen product/query recipe differs")
        binding = DenseBinding(
            metadata["artifact_id"],
            metadata["origin"]["catalog_sha256"],
            digest(list(zip(lexical.keys, lexical.titles, strict=True))),
            metadata["recipe_id"],
            model_identity(encoder.recipe),
        )
        context = FeatureContext(lexical, dense, encoder, binding)
        if "ranker" in paths:
            from atlasrank.ranking.learned import XGBoostRanker

            ranking = RankingRuntime(
                context,
                XGBoostRanker.load(
                    paths["ranker"],
                    context.provenance,
                    reference_query_recipe=reference_query_recipe,
                    runtime_query_recipe=encoder.recipe if reference_query_recipe else None,
                ),
            )
    if peak_rss_bytes() > spec.maximum_rss_bytes:
        raise ValueError("Serving startup exceeds RSS ceiling")
    return PipelineRuntime(
        lexical, identity, tuple(spec.pipelines), context, ranking, spec.maximum_rss_bytes
    )
