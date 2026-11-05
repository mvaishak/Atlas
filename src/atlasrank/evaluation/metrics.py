"""Linear ESCI gains, unknown-label accounting, and deterministic score ordering."""

import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass

ProductKey = tuple[str, str]
GAINS: Mapping[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


@dataclass(frozen=True)
class QueryMetrics:
    ndcg_at_10: float
    mrr_at_10: float
    known_exact_recall_at_20: float | None
    known_exact_recall_at_100: float | None
    judgment_coverage_at_10: float
    judgment_coverage_at_100: float
    returned_at_10: int
    returned_at_100: int
    judged_exact_count: int
    zero_idcg: bool


def rank_scores(scores: Sequence[tuple[ProductKey, float]]) -> list[ProductKey]:
    keys = [key for key, _ in scores]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate scored product")
    if any(not math.isfinite(score) for _, score in scores):
        raise ValueError("Scores must be finite")
    return [key for key, _ in sorted(scores, key=lambda entry: (-entry[1], entry[0]))]


def score_labeled_candidates(
    judgments: Mapping[ProductKey, str],
    scores: Sequence[tuple[ProductKey, float]],
    catalog: Set[ProductKey],
) -> QueryMetrics:
    ranking = rank_scores(scores)
    if set(ranking) != set(judgments):
        raise ValueError("Track A requires every judged candidate and no unjudged candidates")
    return score_query(judgments, ranking, catalog)


def assert_candidate_membership(
    reference: Sequence[ProductKey], reordered: Sequence[ProductKey]
) -> None:
    if (
        len(set(reference)) != len(reference)
        or len(set(reordered)) != len(reordered)
        or set(reference) != set(reordered)
    ):
        raise ValueError("Fixed-pool reranking must preserve unique candidate membership")


def score_query(
    judgments: Mapping[ProductKey, str], ranking: Sequence[ProductKey], catalog: Set[ProductKey]
) -> QueryMetrics:
    """Unknown judgments contribute zero observed gain, preserving their ranks.

    The IDCG always uses the full judged pool. No test-file loader is provided.
    """
    if any(label not in GAINS for label in judgments.values()):
        raise ValueError("Invalid ESCI relevance label")
    if not set(judgments).issubset(catalog) or not set(ranking).issubset(catalog):
        raise ValueError("Unknown product identity")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate product identity in ranking")
    ideal = sorted((GAINS[label] for label in judgments.values()), reverse=True)[:10]
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    dcg = sum(
        GAINS[judgments[key]] / math.log2(i + 2) if key in judgments else 0.0
        for i, key in enumerate(ranking[:10])
    )
    exact = {key for key, label in judgments.items() if label == "E"}
    first = next((i + 1 for i, key in enumerate(ranking[:10]) if key in exact), None)

    def coverage(k: int) -> float:
        returned = ranking[:k]
        return sum(key in judgments for key in returned) / len(returned) if returned else 0.0

    return QueryMetrics(
        ndcg_at_10=dcg / idcg if idcg else 0.0,
        mrr_at_10=1.0 / first if first else 0.0,
        known_exact_recall_at_20=len(exact.intersection(ranking[:20])) / len(exact)
        if exact
        else None,
        known_exact_recall_at_100=(
            len(exact.intersection(ranking[:100])) / len(exact) if exact else None
        ),
        judgment_coverage_at_10=coverage(10),
        judgment_coverage_at_100=coverage(100),
        returned_at_10=min(len(ranking), 10),
        returned_at_100=min(len(ranking), 100),
        judged_exact_count=len(exact),
        zero_idcg=idcg == 0,
    )


def macro_average(results: Sequence[QueryMetrics]) -> dict[str, float | int | None]:
    """Equal official-query weights, including zero-IDCG and empty-return queries."""
    count = len(results)
    eligible = [r for r in results if r.judged_exact_count]
    output: dict[str, float | int | None] = {
        "queries": count,
        "zero_idcg_queries": sum(r.zero_idcg for r in results),
        "recall_queries": len(eligible),
        "recall_excluded_queries": count - len(eligible),
    }
    for field in ("ndcg_at_10", "mrr_at_10", "judgment_coverage_at_10", "judgment_coverage_at_100"):
        values = [float(getattr(r, field)) for r in results]
        output[field] = sum(values) / count if count else None
    for field in ("known_exact_recall_at_20", "known_exact_recall_at_100"):
        values = [float(getattr(r, field)) for r in eligible]
        output[field] = sum(values) / len(values) if values else None
    output["returned_at_10_total"] = sum(r.returned_at_10 for r in results)
    output["returned_at_100_total"] = sum(r.returned_at_100 for r in results)
    return output
