"""Prepare private, deterministic development-only M8 requests without loading models."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from atlasrank.data.splits import normalize_query
from atlasrank.retrieval.analyzer import analyze


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def query_digest(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()


def development_queries(development: Path, audit: Path) -> tuple[pl.DataFrame, str]:
    """Only project the approved query columns after verifying the development pin."""
    if development.name != "development.parquet" or development.resolve().name != development.name:
        raise ValueError("Only an explicit development.parquet input is permitted")
    expected = json.loads(audit.read_text())["output_sha256"]["development.parquet"]
    actual = sha256(development)
    if actual != expected:
        raise ValueError("Development parquet SHA256 mismatch")
    return pl.read_parquet(development, columns=["query_id", "query"]), actual


def query_identities(frame: pl.DataFrame) -> dict[int, str]:
    identities: dict[int, str] = {}
    for query_id, query in frame.select("query_id", "query").iter_rows():
        if type(query_id) is not int or not isinstance(query, str):
            raise ValueError("Query identity requires an integer ID and string query")
        if query_id in identities and identities[query_id] != query:
            raise ValueError("One query ID maps to multiple query strings")
        identities[query_id] = query
    return identities


def select_queries(
    frame: pl.DataFrame,
    count: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate normalized groups using minimum ID, then hash-order with seed 42.

    Limits apply to the normalized string actually sent to HTTP. Exclusion counters
    are mutually exclusive and count unique query IDs, never product rows.
    """
    if count < 1:
        raise ValueError("Query count must be positive")
    identities = query_identities(frame)
    excluded: Counter[str] = Counter(
        {
            "empty": 0,
            "characters_over_512": 0,
            "analyzer_tokens_outside_1_128": 0,
            "duplicate_normalized_group": 0,
        }
    )
    groups: dict[str, int] = {}
    for query_id, raw in sorted(identities.items()):
        query = normalize_query(raw)
        if not query:
            excluded["empty"] += 1
        elif len(query) > 512:
            excluded["characters_over_512"] += 1
        elif not 1 <= len(analyze(query)) <= 128:
            excluded["analyzer_tokens_outside_1_128"] += 1
        elif query in groups:
            excluded["duplicate_normalized_group"] += 1
        else:
            groups[query] = query_id
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            hashlib.sha256(f"42\0{item[0]}\0{item[1]}".encode()).hexdigest(),
            item[1],
        ),
    )
    if len(ordered) < count:
        raise ValueError(f"Insufficient eligible normalized query groups: {len(ordered)} < {count}")
    selected = [
        {"query_id": query_id, "query": query, "query_sha256": query_digest(query)}
        for query, query_id in ordered[:count]
    ]
    return selected, {
        "source_unique_query_ids": len(identities),
        "eligible_normalized_groups": len(groups),
        **excluded,
    }


def write_fresh_json(output: Path, value: object) -> str:
    """Use exclusive creation; never overwrite historical/private evidence."""
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(encoded)
    output.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, source_hash = development_queries(args.development, args.audit)
    queries, excluded = select_queries(frame)
    manifest = {
        "schema_version": 1,
        "partition": "development",
        "seed": 42,
        "source_parquet_sha256": source_hash,
        "audit_sha256": sha256(args.audit),
        "query_count": 1000,
        "queries": queries,
        "exclusion_counts": excluded,
        "selection": "sha256(42\\0normalized_query\\0query_id); minimum ID per group",
        "warmup_requests": 100,
        "measured_requests": 1000,
        "warmup_policy": ("first 100 workload requests; same ordered 1000 measured per trial"),
    }
    digest = write_fresh_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "sha256": digest, "query_count": 1000}))


if __name__ == "__main__":
    main()
