"""Optional fixed-candidate prediction orchestration in the existing application."""

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
        batch = build_features(self.context, query, candidates)
        predicted = self.ranker.predict(batch)
        scores = {key: float(value) for key, value in zip(batch.keys, predicted, strict=True)}
        ordered = rank_scores(list(scores.items()))
        assert_candidate_membership(candidates, ordered)
        return [(key, scores[key]) for key in ordered]
