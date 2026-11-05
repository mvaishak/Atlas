"""Development-query slice membership; no labels are inferred for unknown products."""

from collections.abc import Mapping

from atlasrank.data.splits import normalize_query
from atlasrank.evaluation.metrics import ProductKey, QueryMetrics, macro_average


def query_slices(
    query: str, judgments: Mapping[ProductKey, str], titles: Mapping[ProductKey, str]
) -> dict[str, str]:
    tokens = set(normalize_query(query).split())
    length = len(normalize_query(query).split())
    exact = [key for key, label in judgments.items() if label == "E"]
    overlap = any(
        tokens.intersection(normalize_query(titles.get(key, "")).split()) for key in exact
    )
    return {
        "query_length": "1" if length <= 1 else ("2-3" if length <= 3 else "4+"),
        "exact_title_overlap": ("nonzero" if overlap else "zero") if exact else "no_judged_E",
        "missing_judged_product_title": (
            "yes" if any(not titles.get(key, "").strip() for key in judgments) else "no"
        ),
    }


def summarize_slices(
    results: Mapping[int, QueryMetrics], memberships: Mapping[int, Mapping[str, str]]
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    if results.keys() != memberships.keys():
        raise ValueError("Slice membership and scored query IDs must match")
    grouped: dict[str, dict[str, list[QueryMetrics]]] = {}
    for query_id, result in results.items():
        for dimension, value in memberships[query_id].items():
            grouped.setdefault(dimension, {}).setdefault(value, []).append(result)
    return {
        dimension: {value: macro_average(rows) for value, rows in values.items()}
        for dimension, values in grouped.items()
    }
