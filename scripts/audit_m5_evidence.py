"""Hash local M5 evidence and compare saved BM25 rankings; never run retrieval.

Writes a NEW lightweight inventory. Does not read any official-test file or
retrieve Drive artifacts. Run from the repository root with --output NEW.json.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from atlasrank.artifacts.dense import file_hash, write_once
from atlasrank.data.prepare import load_judgments
from atlasrank.evaluation.m5 import read_rankings, validate_development


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads(Path("configs/m5-reproduction.json").read_text())
    frame = load_judgments(Path("data/esci/prepared-v4"), "development")
    groups = validate_development(frame)
    m4_path = Path("data/esci/reports/m4-cpu-eval-v1/returned/development/per-query.jsonl")
    m4 = {int(r["query_id"]): r for r in map(json.loads, m4_path.read_text().splitlines())}
    bm25_path = Path(".local/colab/m5-catalog-encoding-v1/bm25-top100.jsonl")
    bm25 = read_rankings(bm25_path)
    if bm25.keys() != m4.keys() or bm25.keys() != groups.keys():
        raise ValueError("Saved BM25 query IDs differ")
    for qid, keys in bm25.items():
        if keys != [tuple(k) for k in m4[qid]["B_rankings"]["bm25"]]:
            raise ValueError("Saved BM25 rankings differ from frozen reference")
    paths = [
        "configs/m5-reproduction.json",
        "docs/EVALUATION.md",
        ".local/evaluate_m5_track_a.py",
        ".local/m5-track-a-results.json",
        ".local/export_m5_bm25.py",
        ".local/colab/AtlasRank_M5_Full_Catalog_Encoding.ipynb",
        ".local/colab/m5-catalog-encoding-v1/m5-encoding-input.zip",
        ".local/colab/atlasrank-full-input-v4/products.parquet",
        ".local/colab/m5-training-return-v1/model-bundle/model/model.safetensors",
        ".local/colab/m5-catalog-encoding-v1/m5-per-query-compact.json",
        ".local/colab/m5-catalog-encoding-v1/m4-vs-m5-paired-bootstrap.json",
        str(bm25_path),
        str(m4_path),
        "data/esci/reports/m4-cpu-eval-v1/returned/development/report.json",
        "data/esci/reports/m4-cpu-eval-v1/colab-results.zip",
        "data/esci/prepared-v4/development.parquet",
        "data/esci/indexes/m3-bm25-v2/manifest.json",
    ]
    paths.extend(
        str(p)
        for p in sorted(Path("src/atlasrank/retrieval").glob("*.py"))
        if not p.name.startswith("m5_")
    )
    paths.extend(
        str(p) for p in sorted(Path("src/atlasrank/evaluation").glob("*.py")) if p.name != "m5.py"
    )
    inventory: dict[str, Any] = {}
    for name in paths:
        p = Path(name)
        inventory[name] = {"available": p.is_file()}
        if p.is_file():
            inventory[name].update(bytes=p.stat().st_size, sha256=file_hash(p))
    notebook_path = Path(".local/colab/AtlasRank_M5_Full_Catalog_Encoding.ipynb")
    notebook = json.loads(notebook_path.read_text())
    cells = [
        {
            "cell": i,
            "source_sha256": hashlib.sha256("".join(c.get("source", [])).encode()).hexdigest(),
            "execution_count": c.get("execution_count"),
        }
        for i, c in enumerate(notebook["cells"])
        if c["cell_type"] == "code"
    ]
    result = {
        "schema_version": 1,
        "milestone": "M5.5",
        "local_artifacts": inventory,
        "development": {
            "rows": frame.height,
            "queries": len(groups),
            "normalized_groups": len(set(groups.values())),
        },
        "bm25_saved_ranking_verification": {
            "exact_identity_and_order_match_to_m4": True,
            "queries": len(bm25),
            "hits": sum(map(len, bm25.values())),
            "short_lists": sum(len(v) < 100 for v in bm25.values()),
            "empty_lists": sum(not v for v in bm25.values()),
            "minimum_hits": min(map(len, bm25.values())),
        },
        "historical_notebook": {
            "path": str(notebook_path),
            "cells": cells,
            "recorded_faiss_version": "1.15.1",
            "recorded_sentence_transformers_version": "6.1.0",
            "recorded_gpu": "NVIDIA L4",
        },
        "external_artifacts_not_independently_read": {
            "private_drive": "MyDrive/AtlasRank/m5/catalog-encoding-v1/",
            "paths": [
                "full-catalog-v1/chunks/",
                "faiss-index-v1/index.faiss",
                "faiss-index-v1/ids.parquet",
                "evaluation-v1/dense-top100.jsonl",
                "evaluation-v1/hybrid-top100.jsonl",
            ],
            "reported_index_pins": pins["historical_index"],
        },
        "shared_chat": {
            "url": "https://chatgpt.com/share/6ae4145f-6610-83e8-bd81-7432c37c9807",
            "retrieved": Path(".local/m5-shared-chat.html").is_file(),
            "role": "manual source/provenance only; not independent native artifact verification",
            "local_private_snapshot": ".local/m5-shared-chat.html",
        },
        "test_judgments_accessed": False,
    }
    write_once(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "development": result["development"],
                "bm25": result["bm25_saved_ranking_verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
