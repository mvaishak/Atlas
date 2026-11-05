"""Offline fixed-candidate prediction entry point, not a full model evaluation."""

from collections.abc import Sequence

from atlasrank.evaluation.metrics import ProductKey, rank_scores
from atlasrank.features.ranking import FeatureContext
from atlasrank.features.ranking import build_features as build_features
from atlasrank.ranking.prediction import DeterministicRanker


def offline_predict(
    context: FeatureContext,
    ranker: DeterministicRanker,
    query: str,
    candidates: Sequence[ProductKey],
) -> list[tuple[ProductKey, float]]:
    batch = build_features(context, query, candidates)
    values = ranker.predict(batch)
    scores = {key: float(value) for key, value in zip(batch.keys, values, strict=True)}
    return [(key, scores[key]) for key in rank_scores(list(scores.items()))]
