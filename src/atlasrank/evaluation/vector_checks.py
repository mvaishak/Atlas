"""Approximate cross-kernel cosine diagnostics, separate from exact feature parity."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def compare_reference_ranking(
    reference: NDArray[np.float32],
    rows: list[int],
    scores: list[float],
    k: int,
) -> dict[str, Any]:
    """Check declared score tolerance and reject ordering errors outside that bound.

    Float32 dot-product kernels can disagree near a tie. Exact ties in the
    actual scorer must still use canonical row order. This helper never changes
    scores/order and isn't used for the M6 exact feature-parity gate.
    """
    if reference.ndim != 1 or not np.isfinite(reference).all():
        raise ValueError("Expected finite reference scores")
    if len(rows) != min(k, len(reference)) or len(scores) != len(rows):
        raise AssertionError("Wrong retrieved count")
    if len(set(rows)) != len(rows) or any(row < 0 or row >= len(reference) for row in rows):
        raise AssertionError("Invalid or duplicate vector row")
    rtol, atol = 1e-5, 1e-6
    actual = np.asarray(scores, dtype=np.float64)
    expected = reference[rows].astype(np.float64)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    for i in range(1, len(rows)):
        if actual[i] > actual[i - 1]:
            raise AssertionError("Actual scores aren't in descending order")
        if actual[i] == actual[i - 1] and rows[i] < rows[i - 1]:
            raise AssertionError("Actual score ties aren't canonical")
    reference_rows = np.lexsort((np.arange(len(reference)), -reference))[: len(rows)].tolist()
    max_inversion_gap = 0.0
    for i, first in enumerate(rows):
        for second in rows[i + 1 :]:
            gap = float(reference[second]) - float(reference[first])
            max_inversion_gap = max(max_inversion_gap, gap)
            bound = 2 * atol + rtol * (abs(float(reference[first])) + abs(float(reference[second])))
            if gap > bound:
                raise AssertionError("Reference ordering differs beyond declared score tolerance")
    missing = set(reference_rows) - set(rows)
    if missing:
        worst = min(float(reference[row]) for row in rows)
        for row in missing:
            best_missing = float(reference[row])
            if best_missing - worst > 2 * atol + rtol * (abs(best_missing) + abs(worst)):
                raise AssertionError("A clearly better reference candidate was omitted")
    return {
        "rtol": rtol,
        "atol": atol,
        "membership_exact": set(rows) == set(reference_rows),
        "order_exact": rows == reference_rows,
        "max_absolute_score_error": float(np.max(np.abs(actual - expected), initial=0)),
        "max_reference_order_inversion_gap": max_inversion_gap,
        "actual_ties_canonical": True,
        "well_separated_reference_order_checked": True,
    }
