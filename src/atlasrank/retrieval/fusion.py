"""Frozen rank-only RRF for judged-pool diagnostics or independent retrieval lists."""

from collections.abc import Sequence

from atlasrank.evaluation.metrics import ProductKey


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ProductKey]],
    k: int = 100,
) -> list[tuple[ProductKey, float]]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("Expected positive integer output count")
    scores: dict[ProductKey, float] = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("A retriever list contains duplicate identities")
        for rank, key in enumerate(ranking, 1):
            if key[0] != "us" or not key[1].strip():
                raise ValueError("Expected nonempty US identities")
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
