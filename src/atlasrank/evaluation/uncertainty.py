"""Paired percentile bootstrap over normalized-query groups, never individual rows."""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedInterval:
    delta: float
    lower: float
    upper: float
    queries: int
    groups: int
    excluded_queries: int
    resamples: int = 2000
    seed: int = 42


def percentile(values: Sequence[float], proportion: float) -> float:
    ordered = sorted(values)
    location = proportion * (len(ordered) - 1)
    lo = math.floor(location)
    hi = math.ceil(location)
    return ordered[lo] + (location - lo) * (ordered[hi] - ordered[lo])


def paired_interval(
    reference: Mapping[int, float | None],
    variant: Mapping[int, float | None],
    group_keys: Mapping[int, str],
) -> PairedInterval:
    if not reference or reference.keys() != variant.keys() or reference.keys() != group_keys.keys():
        raise ValueError("Paired query IDs and group membership must match exactly")
    groups: dict[str, list[float]] = {}
    excluded = 0
    for query_id in sorted(reference):
        a, b = reference[query_id], variant[query_id]
        if (a is None) != (b is None):
            raise ValueError("Paired metric eligibility must match")
        if a is None or b is None:
            excluded += 1
            continue
        if not math.isfinite(a) or not math.isfinite(b) or not group_keys[query_id]:
            raise ValueError("Finite scores and nonempty group keys required")
        groups.setdefault(group_keys[query_id], []).append(b - a)
    if not groups:
        raise ValueError("No eligible query pairs")
    grouped = [groups[key] for key in sorted(groups)]
    summaries = [(sum(values), len(values)) for values in grouped]
    rng = random.Random(42)
    sampled = []
    for _ in range(2000):
        draw = rng.choices(summaries, k=len(summaries))
        sampled.append(sum(total for total, _ in draw) / sum(count for _, count in draw))
    queries = sum(count for _, count in summaries)
    return PairedInterval(
        sum(total for total, _ in summaries) / queries,
        percentile(sampled, 0.025),
        percentile(sampled, 0.975),
        queries,
        len(groups),
        excluded,
    )
