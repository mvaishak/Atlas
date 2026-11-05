"""Development-only historical M5 replay and paired evidence; no neural imports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from atlasrank.data.splits import normalize_query
from atlasrank.evaluation.metrics import GAINS, ProductKey, macro_average, score_query
from atlasrank.evaluation.uncertainty import paired_interval
from atlasrank.retrieval.fusion import reciprocal_rank_fusion


def validate_development(frame: pl.DataFrame) -> dict[int, str]:
    required = {"query_id", "query", "product_locale", "product_id", "esci_label", "partition"}
    if (
        not required.issubset(frame.columns)
        or not frame.height
        or any(frame.select(sorted(required)).null_count().row(0))
        or set(frame["partition"]) != {"development"}
        or set(frame["product_locale"]) != {"us"}
        or not set(frame["esci_label"]).issubset(GAINS)
        or frame.unique(["query_id", "product_locale", "product_id"]).height != frame.height
    ):
        raise ValueError("Expected unique development-only US judgments")
    queries = frame.select("query_id", "query").unique().sort("query_id")
    if queries["query_id"].n_unique() != queries.height:
        raise ValueError("One text per official query ID required")
    return {int(q): normalize_query(str(t)) for q, t in queries.iter_rows()}


def read_rankings(path: Path) -> dict[int, list[ProductKey]]:
    output: dict[int, list[ProductKey]] = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            qid = int(row["query_id"])
            hits = row.get("top_100", row.get("top100"))
            if qid in output or not isinstance(hits, list) or len(hits) > 100:
                raise ValueError("Duplicate query or invalid top-100 list")
            keys = [(str(h["product_locale"]), str(h["product_id"])) for h in hits]
            if len(keys) != len(set(keys)):
                raise ValueError("Duplicate ranked product")
            output[qid] = keys
    return output


def evaluate_rankings(
    development: pl.DataFrame,
    bm25: Mapping[int, Sequence[ProductKey]],
    dense: Mapping[int, Sequence[ProductKey]],
    catalog: Set[ProductKey],
) -> tuple[dict[int, Any], dict[str, Any]]:
    groups = validate_development(development)
    if groups.keys() != bm25.keys() or groups.keys() != dense.keys():
        raise ValueError("Matched development query IDs required")
    per_query: dict[int, Any] = {}
    aggregate: dict[str, list[Any]] = {name: [] for name in ["bm25", "dense", "hybrid"]}
    for (qid,), frame in development.sort("query_id").group_by("query_id", maintain_order=True):
        qid = int(qid)
        judgments = {
            (str(loc), str(pid)): str(label)
            for loc, pid, label in frame.select(
                "product_locale", "product_id", "esci_label"
            ).iter_rows()
        }
        if len(bm25[qid]) > 100 or len(dense[qid]) > 100:
            raise ValueError("Independent top-100 retrieval lists required")
        rankings = {
            "bm25": bm25[qid],
            "dense": dense[qid],
            "hybrid": [key for key, _ in reciprocal_rank_fusion([bm25[qid], dense[qid]], k=100)],
        }
        per_query[qid] = {}
        for name, ranking in rankings.items():
            metric = score_query(judgments, ranking, catalog)
            aggregate[name].append(metric)
            per_query[qid][name] = asdict(metric)
    names = {
        "queries",
        "recall_queries",
        "recall_excluded_queries",
        "known_exact_recall_at_20",
        "known_exact_recall_at_100",
        "judgment_coverage_at_10",
        "judgment_coverage_at_100",
        "returned_at_10_total",
        "returned_at_100_total",
    }
    return per_query, {
        name: {k: v for k, v in macro_average(values).items() if k in names}
        for name, values in aggregate.items()
    }


def compare_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [int(r["query_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate paired query ID")
    a = {int(r["query_id"]): r["reference"] for r in rows}
    b = {int(r["query_id"]): r["variant"] for r in rows}
    groups = {int(r["query_id"]): r["group"] for r in rows}
    interval = paired_interval(a, b, groups)
    eligible = [q for q in ids if a[q] is not None and b[q] is not None]
    return {
        "m4_mean": sum(a[q] for q in eligible) / len(eligible),
        "m5_mean": sum(b[q] for q in eligible) / len(eligible),
        "paired_interval": asdict(interval),
        "queries_improved": sum(b[q] - a[q] > 1e-12 for q in eligible),
        "queries_worsened": sum(b[q] - a[q] < -1e-12 for q in eligible),
        "queries_unchanged": sum(abs(b[q] - a[q]) <= 1e-12 for q in eligible),
    }


def track_a(
    development: pl.DataFrame, products: pl.DataFrame, encoder: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Historical NumPy float32 candidate scoring; no native FAISS in this process."""
    import numpy as np

    from atlasrank.artifacts.dense import product_keys
    from atlasrank.evaluation.metrics import rank_scores, score_labeled_candidates

    validate_development(development)
    catalog = frozenset(product_keys(products))
    joined = development.join(products, on=["product_locale", "product_id"], how="left")
    if joined.height != development.height or joined["product_title"].null_count():
        raise ValueError("Invalid development/title join")
    rows, metrics = [], []
    for frame in joined.sort("query_id", "product_locale", "product_id").partition_by(
        "query_id", maintain_order=True
    ):
        keys = list(frame.select("product_locale", "product_id").iter_rows())
        judgments = dict(zip(keys, frame["esci_label"].to_list(), strict=True))
        query = encoder.encode([str(frame["query"][0])])[0]
        titles = frame["product_title"].to_list()
        vectors = np.concatenate(
            [encoder.encode(titles[i : i + 32]) for i in range(0, len(titles), 32)]
        )
        scores = vectors @ query
        scored = [(key, float(value)) for key, value in zip(keys, scores, strict=True)]
        metric = score_labeled_candidates(judgments, scored, catalog)
        metrics.append(metric)
        rows.append(
            {
                "query_id": int(frame["query_id"][0]),
                "ranking": rank_scores(scored),
                "metrics": asdict(metric),
            }
        )
    summary = macro_average(metrics)
    return rows, {
        k: summary[k] for k in ["queries", "zero_idcg_queries", "ndcg_at_10", "mrr_at_10"]
    }


def export_bm25(development: pl.DataFrame, lexical: Any, output: Path) -> dict[str, int]:
    validate_development(development)
    total = 0
    queries = development.select("query_id", "query").unique().sort("query_id")
    with output.open("x") as target:
        for qid, query in queries.iter_rows():
            hits = lexical.hits_from_scores(lexical.score_all(query), 100)
            total += len(hits)
            target.write(
                json.dumps(
                    {
                        "query_id": int(qid),
                        "query": query,
                        "top_100": [
                            {"product_locale": h.product_locale, "product_id": h.product_id}
                            for h in hits
                        ],
                    }
                )
                + "\n"
            )
    return {"queries": queries.height, "returned_hits": total}
