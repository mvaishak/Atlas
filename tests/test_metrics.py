"""Human-auditable linear-gain checks and adversarial evaluator boundaries."""

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from atlasrank.evaluation.metrics import (
    assert_candidate_membership,
    macro_average,
    rank_scores,
    score_labeled_candidates,
    score_query,
)
from atlasrank.evaluation.slices import query_slices, summarize_slices
from atlasrank.evaluation.uncertainty import paired_interval
from atlasrank.evaluation.xgboost_check import conformance


def test_frozen_metric_fixtures_and_independent_decimal_reference() -> None:
    fixtures = json.loads((Path(__file__).parent / "fixtures/metrics.json").read_text())
    for name in ("reverse", "unknown_before_exact"):
        case = fixtures[name]
        judgments = {("us", key): label for key, label in case["judgments"].items()}
        ranking = [("us", key) for key in case["ranking"]]
        result = score_query(judgments, ranking, set(judgments).union(ranking))
        assert result.ndcg_at_10 == pytest.approx(case["expected_ndcg_at_10"], abs=1e-14)
        assert result.mrr_at_10 == case["expected_mrr_at_10"]
        assert result.known_exact_recall_at_20 == case["expected_known_exact_recall"]
        assert result.judgment_coverage_at_10 == case["expected_coverage"]
    with localcontext() as context:
        context.prec = 60
        log3 = Decimal(3).ln() / Decimal(2).ln()
        log5 = Decimal(5).ln() / Decimal(2).ln()
        reverse = (Decimal("0.01") / log3 + Decimal("0.1") / 2 + 1 / log5) / (
            1 + Decimal("0.1") / log3 + Decimal("0.01") / 2
        )
        assert float(reverse) == pytest.approx(
            fixtures["reverse"]["expected_ndcg_at_10"], abs=1e-15
        )


def test_cutoffs_full_judged_denominators_and_short_returns() -> None:
    catalog = {("us", str(i)) for i in range(105)}
    ranking = [("us", str(i)) for i in range(101)]
    result = score_query({("us", "20"): "E", ("us", "100"): "E"}, ranking, catalog)
    assert result.ndcg_at_10 == 0
    assert result.mrr_at_10 == 0
    assert result.known_exact_recall_at_20 == 0
    assert result.known_exact_recall_at_100 == 0.5
    assert result.judgment_coverage_at_100 == 0.01
    short = score_query({("us", "1"): "E"}, [("us", "1")], catalog)
    assert short.judgment_coverage_at_10 == 1
    assert short.returned_at_10 == 1


def test_zero_idcg_no_exact_empty_rankings_and_macro_denominators() -> None:
    key = ("us", "a")
    zero = score_query({key: "I"}, [key], {key})
    assert zero.zero_idcg and zero.ndcg_at_10 == zero.mrr_at_10 == 0
    assert zero.known_exact_recall_at_20 is None
    empty = score_query({key: "E"}, [], {key})
    assert empty.ndcg_at_10 == empty.known_exact_recall_at_20 == 0
    mean = macro_average([zero, empty])
    assert mean["queries"] == 2
    assert mean["recall_queries"] == mean["recall_excluded_queries"] == 1
    assert mean["zero_idcg_queries"] == 1
    assert mean["judgment_coverage_at_10"] == 0.5
    assert macro_average([])["ndcg_at_10"] is None


def test_invalid_identity_label_and_ties() -> None:
    a, b = ("us", "a"), ("us", "b")
    for ranking in ([a, a], [b]):
        with pytest.raises(ValueError):
            score_query({a: "E"}, ranking, {a})
    with pytest.raises(ValueError):
        score_query({a: "invalid"}, [a], {a})
    with pytest.raises(ValueError):
        score_query({b: "E"}, [], {a})
    assert rank_scores([(b, 1), (a, 1)]) == [a, b]
    for scores in ([(a, math.nan)], [(a, math.inf)], [(a, 1), (a, 2)]):
        with pytest.raises(ValueError):
            rank_scores(scores)


def test_track_a_complete_candidates_and_fixed_pool_contract() -> None:
    a, b = ("us", "a"), ("us", "b")
    result = score_labeled_candidates({a: "E", b: "I"}, [(b, 1.0), (a, 1.0)], {a, b})
    assert result.ndcg_at_10 == 1
    with pytest.raises(ValueError, match="every judged"):
        score_labeled_candidates({a: "E", b: "I"}, [(a, 1.0)], {a, b})
    assert_candidate_membership([a, b], [b, a])
    for invalid in ([a], [a, a]):
        with pytest.raises(ValueError):
            assert_candidate_membership([a, b], invalid)


def test_grouped_paired_bootstrap_is_deterministic_and_counts_queries() -> None:
    a = {1: 0.0, 2: 0.0, 3: 0.0, 4: None}
    b = {1: 0.1, 2: 0.1, 3: 0.9, 4: None}
    groups = {1: "same", 2: "same", 3: "different", 4: "excluded"}
    result = paired_interval(a, b, groups)
    assert result == paired_interval(dict(reversed(list(a.items()))), b, groups)
    assert result.queries == 3 and result.groups == 2 and result.excluded_queries == 1
    assert result.delta == pytest.approx(1.1 / 3)
    assert result.lower == pytest.approx(0.1)
    assert result.upper == pytest.approx(0.9)
    assert result.resamples == 2000 and result.seed == 42
    with pytest.raises(ValueError):
        paired_interval(a, {1: 0.0}, groups)
    with pytest.raises(ValueError):
        paired_interval({1: None}, {1: 0.0}, {1: "x"})


def test_development_slice_membership() -> None:
    a, b = ("us", "a"), ("us", "b")
    assert query_slices(" ＲＥＤ  shoe ", {a: "E", b: "I"}, {a: "red shoes", b: ""}) == {
        "query_length": "2-3",
        "exact_title_overlap": "nonzero",
        "missing_judged_product_title": "yes",
    }
    result = summarize_slices(
        {1: score_query({a: "E"}, [a], {a}), 2: score_query({a: "I"}, [], {a})},
        {1: {"length": "1"}, 2: {"length": "1"}},
    )
    assert result["length"]["1"]["queries"] == 2
    assert result["length"]["1"]["recall_excluded_queries"] == 1


def test_pinned_xgboost_linear_gains_without_updates() -> None:
    result = conformance()
    assert result["boosted_rounds"] == 0
    assert result["default_exponential_gain_rejected"] is True
