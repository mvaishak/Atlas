"""Replay frozen development metrics; do not infer, rank, or read judgment columns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from prepare_m8_workload import (
    development_queries,
    query_identities,
    sha256,
    write_fresh_json,
)

from atlasrank.data.splits import normalize_query


def frozen_group_digest(query: str) -> str:
    """Match data.splits.assign_queries: seed 42, US locale, normalized canonical text."""
    return hashlib.sha256(f"42|us|{normalize_query(query)}".encode()).hexdigest()


def summarize_rows(
    rows: list[dict[str, Any]],
    identities: dict[int, str],
    expected_count: int = 4129,
) -> dict[str, Any]:
    if len(rows) != expected_count or len(identities) != expected_count:
        raise ValueError("Frozen development query count mismatch")
    seen: set[int] = set()
    groups: set[str] = set()
    deltas: list[dict[str, Any]] = []
    for row in rows:
        query_id = row["query_id"]
        if type(query_id) is not int or query_id in seen or query_id not in identities:
            raise ValueError("Invalid, duplicated, or non-development query ID")
        group = frozen_group_digest(identities[query_id])
        if row["group_sha256"] != group or group in groups:
            raise ValueError("Frozen normalized query group mismatch or duplicate")
        if row["candidate_membership_exact"] is not True:
            raise ValueError("Frozen candidate membership was not exact")
        before = row["metrics"]["C_reference"]["ndcg_at_10"]
        after = row["metrics"]["C_ranker"]["ndcg_at_10"]
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in (before, after)
        ):
            raise ValueError("Invalid frozen NDCG metric")
        tokens = len(normalize_query(identities[query_id]).split())
        if tokens < 1:
            raise ValueError("Frozen development query has no normalized whitespace tokens")
        bucket = "1" if tokens == 1 else "2-3" if tokens <= 3 else "4+"
        deltas.append(
            {
                "query_id": query_id,
                "query_length_slice": bucket,
                "C_reference_ndcg_at_10": before,
                "C_ranker_ndcg_at_10": after,
                "delta_ndcg_at_10": after - before,
            }
        )
        seen.add(query_id)
        groups.add(group)
    if seen != identities.keys():
        raise ValueError("Frozen query IDs differ from development identities")
    slices: dict[str, Any] = {}
    for bucket in ("1", "2-3", "4+"):
        selected = [row for row in deltas if row["query_length_slice"] == bucket]
        n = len(selected)
        slices[bucket] = {
            "queries": n,
            "excluded_queries": 0,
            "mean_delta_ndcg_at_10": (
                math.fsum(row["delta_ndcg_at_10"] for row in selected) / n if n else None
            ),
        }
    ordered = sorted(deltas, key=lambda row: (row["delta_ndcg_at_10"], row["query_id"]))
    return {
        "queries": len(rows),
        "excluded_queries": 0,
        "mean_C_delta_ndcg_at_10": math.fsum(row["delta_ndcg_at_10"] for row in deltas) / len(rows),
        "query_length_definition": "EVALUATION.md normalized whitespace token count",
        "query_length_slices": slices,
        "worst_10_query_ids": ordered[:10],
        "best_10_query_ids": sorted(
            deltas, key=lambda row: (-row["delta_ndcg_at_10"], row["query_id"])
        )[:10],
    }


def build_summary(
    development: Path,
    audit: Path,
    m4_report: Path,
    m6_report: Path,
    per_query: Path,
) -> dict[str, Any]:
    frame, source_hash = development_queries(development, audit)
    m4 = json.loads(m4_report.read_text())
    m6 = json.loads(m6_report.read_text())
    if m4["partition"] != "development" or m6["test_partition"] != "locked_not_evaluated":
        raise ValueError("Reports must represent saved development evaluation with locked test")
    if m6["development_queries"] != 4129:
        raise ValueError("Unexpected frozen development count")
    if m4["provenance"]["source"]["development_sha256"] != source_hash:
        raise ValueError("M4 development source pin mismatch")
    if m4["provenance"]["source"]["prepared_audit_sha256"] != sha256(audit):
        raise ValueError("M4 audit source pin mismatch")
    if sha256(per_query) != m6["evaluation"]["per_query_sha256"]:
        raise ValueError("M6 per-query source pin mismatch")
    with per_query.open() as stream:
        rows = [json.loads(line) for line in stream]
    replay = summarize_rows(rows, query_identities(frame))
    c = m6["evaluation"]["C"]
    expected_delta = c["ranker"]["ndcg_at_10"] - c["same_pool_rrf"]["ndcg_at_10"]
    if not math.isclose(replay["mean_C_delta_ndcg_at_10"], expected_delta, abs_tol=1e-12):
        raise ValueError("Per-query aggregate disagrees with frozen report")
    return {
        "schema_version": 1,
        "partition": "development",
        "status": "saved_development_replay_not_new_inference",
        "sources_sha256": {
            "development_parquet": source_hash,
            "audit": sha256(audit),
            "m4_report": sha256(m4_report),
            "m6_report": sha256(m6_report),
            "m6_per_query": sha256(per_query),
        },
        "m4_original_report": m4,
        "m6_original_report": m6,
        "error_analysis": replay,
        "unknown_judgment_caveat": m6["evaluation"]["unknown_policy"],
        "test_partition": "locked_not_evaluated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "audit", "m4-report", "m6-report", "per-query", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(
        args.development, args.audit, args.m4_report, args.m6_report, args.per_query
    )
    digest = write_fresh_json(args.output, summary)
    print(json.dumps({"output": str(args.output), "sha256": digest, "status": summary["status"]}))


if __name__ == "__main__":
    main()
