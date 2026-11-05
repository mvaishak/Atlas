"""Optional fixed-candidate prediction orchestration in the existing application."""

import time
from dataclasses import dataclass

from atlasrank.evaluation.metrics import ProductKey, assert_candidate_membership, rank_scores
from atlasrank.features.ranking import FeatureContext
from atlasrank.features.ranking import build_features as build_features
from atlasrank.ranking.prediction import DeterministicRanker
from atlasrank.retrieval.bm25 import LexicalIndex


@dataclass
class RankingRuntime:
    context: FeatureContext
    ranker: DeterministicRanker

    def validate(self, lexical: LexicalIndex) -> None:
        self.context.validate()
        self.ranker.validate(self.context.provenance)
        if lexical.bundle_id != self.context.provenance.lexical_bundle_id:
            raise ValueError("Serving lexical bundle differs from ranker context")

    def predict(self, query: str, candidates: list[ProductKey]) -> list[tuple[ProductKey, float]]:
        ranked, _ = self.predict_with_timings(query, candidates)
        return ranked

    def predict_with_timings(
        self, query: str, candidates: list[ProductKey]
    ) -> tuple[list[tuple[ProductKey, float]], dict[str, float]]:
        """Time shared feature construction, prediction, and final ordering.

        Feature construction includes build_features' own query encoding and
        candidate scoring. It is distinct from retrieval's initial encoding.
        """
        timings: dict[str, float] = {}
        started = time.perf_counter()
        batch = build_features(self.context, query, candidates)
        timings["feature_construction"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        predicted = self.ranker.predict(batch)
        timings["ranker_prediction"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        scores = {key: float(value) for key, value in zip(batch.keys, predicted, strict=True)}
        ordered = rank_scores(list(scores.items()))
        assert_candidate_membership(candidates, ordered)
        ranked = [(key, scores[key]) for key in ordered]
        timings["ranking_order"] = (time.perf_counter() - started) * 1000
        return ranked, timings
