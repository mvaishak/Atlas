"""Fixed supervised ranker contract; no native ML imports or test partition loader."""

import copy
from typing import Any

import polars as pl

from atlasrank.data.splits import normalize_query

PARAMS: dict[str, Any] = {
    "objective": "rank:ndcg",
    "ndcg_exp_gain": False,
    "tree_method": "hist",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "min_child_weight": 1,
    "seed": 42,
    "device": "cpu",
    "nthread": 1,
}
GRADES = {"E": 100, "S": 10, "C": 1, "I": 0}


def require_fit_budget(remaining_seconds: float, evaluation_reserve_seconds: float) -> None:
    if evaluation_reserve_seconds <= 0 or remaining_seconds <= evaluation_reserve_seconds + 60:
        raise RuntimeError("Insufficient time for fixed fit plus measured evaluation reserve")


def validate_baseline(
    pins: dict[str, Any],
    frozen_files: dict[str, str],
    complete: dict[str, Any],
    report: dict[str, Any],
) -> None:
    for name in ("report.json", "COMPLETE.json", "per-query.jsonl"):
        if pins["baseline/" + name]["sha256"] != frozen_files["development/" + name]:
            raise ValueError(f"Frozen baseline file differs: {name}")
    if (
        complete["report_sha256"] != pins["baseline/report.json"]["sha256"]
        or report["per_query_sha256"] != pins["baseline/per-query.jsonl"]["sha256"]
    ):
        raise ValueError("Frozen baseline cross-reference differs")


def fixed_configuration(config: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = {"params": PARAMS, "num_boost_round": 100}
    if config is not None and any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Only the approved fixed XGBoost experiment is allowed")
    return copy.deepcopy(expected if config is None else config)


def validate_groups(train: pl.DataFrame, development: pl.DataFrame) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    identities: dict[str, dict[str, set[Any]]] = {}
    required = {
        "query_id",
        "query",
        "group_sha256",
        "partition",
        "product_locale",
        "product_id",
        "esci_label",
    }
    for name, frame in (("train", train), ("development", development)):
        if not required.issubset(frame.columns) or not frame.height:
            raise ValueError("Complete nonempty supervised groups required")
        if any(frame[field].null_count() for field in required):
            raise ValueError("Null supervised group field")
        if set(frame["partition"]) != {name} or set(frame["product_locale"]) != {"us"}:
            raise ValueError("Supervised partition or locale differs")
        labels = frame["esci_label"].value_counts()
        counts = dict(labels.iter_rows())
        if not set(counts).issubset(GRADES) or (name == "train" and set(counts) != set(GRADES)):
            raise ValueError("All four supervised ESCI grades required; unknown grade rejected")
        if frame.unique(subset=["query_id", "product_locale", "product_id"]).height != frame.height:
            raise ValueError("Duplicate supervised query/product pair")
        queries = frame.select("query_id", "query", "group_sha256").unique()
        if queries["query_id"].n_unique() != queries.height:
            raise ValueError("Query ID maps to inconsistent text/group")
        normalized = {normalize_query(str(text)) for text in queries["query"]}
        group_keys = set(queries["group_sha256"])
        if (
            "" in normalized
            or "" in group_keys
            or any(not str(p).strip() for p in frame["product_id"])
        ):
            raise ValueError("Empty supervised identity")
        identities[name] = {
            "query_id": set(queries["query_id"]),
            "normalized_text": normalized,
            "group": group_keys,
        }
        summaries[name] = {"rows": frame.height, "groups": len(group_keys), "labels": counts}
    for field in identities["train"]:
        if identities["train"][field] & identities["development"][field]:
            raise ValueError(f"Training/development overlap: {field}")
    return summaries
