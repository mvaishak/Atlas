"""Hand-checked report extraction with private-data-free synthetic sources."""

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from atlasrank.artifacts.dense import digest, file_hash
from atlasrank.retrieval.bm25 import build
from atlasrank.serving.comparison import REPO, REVISION


def inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    prepared = root / "prepared"
    prepared.mkdir()
    dev = pl.DataFrame(
        {
            "query_id": [q for q in range(1, 6) for _ in range(2)],
            "query": [f"query {q}" for q in range(1, 6) for _ in range(2)],
            "product_locale": ["us"] * 10,
            "product_id": ["a", "b"] * 5,
            "esci_label": ["E", "I"] * 5,
            "partition": ["development"] * 10,
            "group_sha256": [str(q) * 64 for q in range(1, 6) for _ in range(2)],
        }
    )
    dev.write_parquet(prepared / "development.parquet")
    audit = {
        "schema_version": 1,
        "evaluation_contract": "v0.3",
        "output_sha256": {"development.parquet": file_hash(prepared / "development.parquet")},
    }
    (prepared / "audit.json").write_text(json.dumps(audit))
    (prepared / "COMPLETE.json").write_text(
        json.dumps({"schema_version": 1, "audit_sha256": file_hash(prepared / "audit.json")})
    )
    source = {
        "catalog_sha256": "c" * 64,
        "development_sha256": file_hash(prepared / "development.parquet"),
        "prepared_audit_sha256": file_hash(prepared / "audit.json"),
    }
    lexical = root / "lexical"
    manifest = build(
        pl.DataFrame(
            {
                "product_locale": ["us"] * 3,
                "product_id": ["a", "b", "c"],
                "product_title": ["apple", "berry", "unjudged catalog product"],
            }
        ),
        lexical,
        source,
    )
    records = root / "per-query.jsonl"
    rows = [
        {
            "query_id": q,
            "group_sha256": str(q) * 64,
            "B_rankings": {
                "bm25": [] if q == 5 else [["us", "a"], ["us", "b"]],
                "hybrid": [["us", "c"], ["us", "a"]],
            },
            "B_scores": {
                "bm25": [] if q == 5 else [[["us", "a"], 3.125], [["us", "b"], 0.125]],
                "hybrid": [[["us", "c"], 0.032], [["us", "a"], 0.02]],
            },
        }
        for q in range(1, 6)
    ]
    records.write_text("".join(json.dumps(r) + "\n" for r in rows))
    recipe = {"repo": REPO, "revision": REVISION}
    report = root / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partition": "development",
                "evaluation_contract": "v0.3",
                "retriever_k": 100,
                "rrf_constant": 60,
                "per_query_sha256": file_hash(records),
                "test_judgments_accessed": False,
                "provenance": {
                    "lexical_artifact_id": manifest["bundle_id"],
                    "dense_artifact_id": "fixture-dense",
                    "product_recipe": recipe,
                    "query_recipe": recipe,
                    "product_recipe_id": digest(recipe),
                    "recipe_id": digest(recipe),
                    "source": source,
                },
            }
        )
    )
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "query_ids": [1, 2, 3, 4, 5],
                "report_sha256": file_hash(report),
                "per_query_sha256": file_hash(records),
                "lexical_manifest_sha256": file_hash(lexical / "manifest.json"),
                "evaluation_contract_sha256": file_hash(Path("docs/EVALUATION.md")),
                "development_sha256": source["development_sha256"],
            }
        )
    )
    return config, prepared, lexical, report, records


def test_extracts_exact_source_order_scores_unknowns_and_partial_lists(tmp_path: Path) -> None:
    from atlasrank.serving.demo_prepare import prepare_examples

    config, prepared, lexical, report, records = inputs(tmp_path)
    output = tmp_path / "demo.json"
    result = prepare_examples(config, prepared, lexical, report, records, output)
    data = json.loads(output.read_text())
    assert result["sha256"] == file_hash(output)
    assert len(data["examples"]) == 5
    ranked = data["examples"][0]["rankings"]
    assert [(h["product_id"], h["score"], h["judgment"]) for h in ranked["bm25"]] == [
        ("a", 3.125, "E"),
        ("b", 0.125, "I"),
    ]
    assert [(h["product_id"], h["score"], h["judgment"]) for h in ranked["hybrid"]] == [
        ("c", 0.032, "unknown"),
        ("a", 0.02, "E"),
    ]
    assert data["examples"][4]["rankings"]["bm25"] == []
    assert data["pipelines"]["hybrid"]["historical_search_ms"] is None
    assert data["sources"]["provenance"] == json.loads(report.read_text())["provenance"]
    with pytest.raises(FileExistsError):
        prepare_examples(config, prepared, lexical, report, records, output)


@pytest.mark.parametrize(
    "fault",
    ["source_hash", "missing_query", "wrong_group", "score_alignment", "missing_title", "train"],
)
def test_corrupt_or_wrong_source_does_not_publish(tmp_path: Path, fault: str) -> None:
    from atlasrank.serving.demo_prepare import prepare_examples

    config, prepared, lexical, report, records = inputs(tmp_path)
    pins: dict[str, Any] = json.loads(config.read_text())
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    if fault == "missing_query":
        rows = rows[:-1]
    if fault == "wrong_group":
        rows[0]["group_sha256"] = "0" * 64
    if fault == "score_alignment":
        rows[0]["B_scores"]["hybrid"][0][0] = ["us", "b"]
    if fault == "missing_title":
        rows[0]["B_rankings"]["hybrid"][0] = ["us", "absent"]
        rows[0]["B_scores"]["hybrid"][0][0] = ["us", "absent"]
    if fault == "train":
        frame = pl.read_parquet(prepared / "development.parquet").with_columns(
            pl.lit("train").alias("partition")
        )
        frame.write_parquet(prepared / "development.parquet")
    records.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pins["per_query_sha256"] = "0" * 64 if fault == "source_hash" else file_hash(records)
    r = json.loads(report.read_text())
    r["per_query_sha256"] = pins["per_query_sha256"]
    report.write_text(json.dumps(r))
    pins["report_sha256"] = file_hash(report)
    config.write_text(json.dumps(pins))
    output = tmp_path / "demo.json"
    with pytest.raises((ValueError, KeyError)):
        prepare_examples(config, prepared, lexical, report, records, output)
    assert not output.exists()
