"""A supervision contract for future samplers; no sampling or loss implementation."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

LABELS = frozenset({"E", "S", "C", "I"})


@dataclass(frozen=True)
class Supervision:
    positives: frozenset[str]
    negatives: frozenset[str]
    usable: bool


def admissible(partition: str, judgments: Mapping[str, str]) -> Supervision:
    if partition != "train":
        raise ValueError("Only training-manifest groups can supply supervision")
    if any(not product or label not in LABELS for product, label in judgments.items()):
        raise ValueError("Invalid product identity or ESCI label")
    positives = frozenset(p for p, label in judgments.items() if label == "E")
    negatives = frozenset(p for p, label in judgments.items() if label == "I")
    return Supervision(positives, negatives, bool(positives and negatives))


def validate_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Collapse identical judgments and fail closed on conflicts before admissibility."""
    result: dict[str, str] = {}
    for product, label in pairs:
        if label not in LABELS or not product:
            raise ValueError("Invalid judgment")
        if product in result and result[product] != label:
            raise ValueError("Conflicting duplicate judgments")
        result[product] = label
    return result
