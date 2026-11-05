"""Evaluate fixed synthetic margins in a zero-tree Booster; never fit/update."""

import json
import math
from typing import Any

import numpy as np
import xgboost as xgb


def conformance() -> dict[str, Any]:
    # Every group is already sorted by ascending synthetic product key. Equal
    # predictions therefore exercise the evaluator's required canonical tie order.
    labels = [0, 1, 10, 100] + [0] * 10 + [100, 0] + [0, 1, 10, 100] + [0, 0]
    margins = [4.0, 3.0, 2.0, 1.0] + list(map(float, range(12, 0, -1))) + [1.0] * 4 + [2.0, 1.0]
    groups = [4, 12, 4, 2]
    matrix = xgb.DMatrix(
        np.zeros((len(labels), 1), dtype=np.float32), label=labels, base_margin=margins, nthread=1
    )
    matrix.set_group(groups)
    booster = xgb.Booster(
        params={
            "objective": "rank:ndcg",
            "ndcg_exp_gain": False,
            "eval_metric": ["ndcg@10", "ndcg@10-"],
            "device": "cpu",
            "nthread": 1,
        },
        cache=[matrix],
    )
    evaluated = booster.eval_set([(matrix, "synthetic")])
    values = {
        entry.split(":")[0]: float(entry.split(":")[1]) for entry in evaluated.split("\t")[1:]
    }
    # Independent fixed hand-calculated result, not a call to AtlasRank's evaluator.
    reverse = 0.45593957346856512
    expected = (reverse + 0.0 + reverse + 0.0) / 4
    observed = values["synthetic-ndcg@10-"]
    if not math.isclose(observed, expected, abs_tol=1e-12):
        raise AssertionError(f"XGBoost linear gain/tie/cutoff mismatch: {observed} vs {expected}")
    if not math.isclose(values["synthetic-ndcg@10"] - observed, 0.25, abs_tol=1e-12):
        raise AssertionError("Unexpected default zero-IDCG behavior")
    config = json.loads(booster.save_config())
    objective = config["learner"]["objective"]
    if (
        objective["name"] != "rank:ndcg"
        or objective["lambdarank_param"]["ndcg_exp_gain"] != "0"
        or booster.num_boosted_rounds() != 0
    ):
        raise AssertionError("Objective, gains, or zero-update contract changed")
    default = xgb.Booster(
        params={"objective": "rank:ndcg", "eval_metric": "ndcg@10", "device": "cpu", "nthread": 1},
        cache=[matrix],
    )
    try:
        default.eval_set([(matrix, "synthetic")])
    except xgb.core.XGBoostError as error:
        if "31" not in str(error) or "exponential" not in str(error):
            raise
        rejected = True
    else:
        rejected = False
    if not rejected:
        raise AssertionError("Default exponential gain must reject labels above 31")
    return {
        "schema_version": 1,
        "xgboost_version": xgb.__version__,
        "device": "cpu",
        "boosted_rounds": booster.num_boosted_rounds(),
        "objective": objective,
        "groups": groups,
        "fixed_labels": labels,
        "fixed_margins": margins,
        "linear_ndcg_at_10_minus": observed,
        "independent_expected": expected,
        "default_zero_idcg_macro_offset": 0.25,
        "zero_idcg_adapter": "Use ndcg@10-; default ndcg@10 assigns 1 to zero-IDCG groups",
        "default_exponential_gain_rejected": rejected,
        "notes": (
            "Synthetic evaluator conformance only; no predictions, fitting, updates, or model saved"
        ),
    }
