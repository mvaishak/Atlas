"""Extract five source-pinned development replays without executing retrieval."""

import json
from pathlib import Path
from typing import Any

import polars as pl

from atlasrank.artifacts.dense import digest, file_hash, write_once
from atlasrank.artifacts.lexical import validate as validate_lexical
from atlasrank.data.prepare import load_judgments
from atlasrank.evaluation.m5 import validate_development
from atlasrank.serving.comparison import MAX_FIXTURE_BYTES, Comparison


def require_hash(path: Path, expected: str) -> None:
    if path.is_symlink() or file_hash(path) != expected:
        raise ValueError("Demo source checksum differs")


def prepare_examples(
    config: Path, prepared: Path, lexical: Path, report_path: Path, records: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve previous demo fixtures")
    pins = json.loads(config.read_text())
    ids = pins["query_ids"]
    if (
        pins["schema_version"] != 1
        or len(ids) != 5
        or len(set(ids)) != 5
        or any(type(q) is not int or q < 0 for q in ids)
    ):
        raise ValueError("Exactly five explicit query IDs required")
    for path, sha in [
        (report_path, pins["report_sha256"]),
        (records, pins["per_query_sha256"]),
        (lexical / "manifest.json", pins["lexical_manifest_sha256"]),
        (prepared / "development.parquet", pins["development_sha256"]),
        (Path("docs/EVALUATION.md"), pins["evaluation_contract_sha256"]),
    ]:
        require_hash(path, sha)
    manifest = validate_lexical(lexical)
    report = json.loads(report_path.read_text())
    if (
        report["partition"] != "development"
        or report["evaluation_contract"] != "v0.3"
        or report["retriever_k"] != 100
        or report["rrf_constant"] != 60
        or report["test_judgments_accessed"] is not False
        or report["per_query_sha256"] != pins["per_query_sha256"]
    ):
        raise ValueError("Frozen M4 development report required")
    provenance = report["provenance"]
    origin = provenance["source"]
    if (
        provenance["lexical_artifact_id"] != manifest["bundle_id"]
        or origin["development_sha256"] != pins["development_sha256"]
        or origin["prepared_audit_sha256"] != file_hash(prepared / "audit.json")
        or any(
            manifest["origin"][key] != origin[key]
            for key in ["catalog_sha256", "prepared_audit_sha256"]
        )
    ):
        raise ValueError("Lexical/catalog/development provenance differs")
    development = load_judgments(prepared, "development")
    validate_development(development)
    selected = development.filter(pl.col("query_id").is_in(ids))
    if set(selected["query_id"]) != set(ids):
        raise ValueError("Selected query missing from development")
    found: dict[int, Any] = {}
    with records.open() as stream:
        for line in stream:
            row = json.loads(line)
            qid = row["query_id"]
            if qid in ids:
                if qid in found:
                    raise ValueError("Duplicate selected source record")
                found[qid] = row
    if set(found) != set(ids):
        raise ValueError("Selected query missing from report")
    keys: set[tuple[str, str]] = set()
    for row in found.values():
        for method in ["bm25", "hybrid"]:
            rankings, scores = row["B_rankings"][method], row["B_scores"][method]
            if (
                len(rankings) != len(scores)
                or len(rankings) > 100
                or len({tuple(k) for k in rankings}) != len(rankings)
                or any(score[0] != key for key, score in zip(rankings, scores, strict=True))
            ):
                raise ValueError("Saved ranking/score alignment differs")
            keys.update(tuple(key) for key in rankings[:10])
    products = (
        pl.scan_parquet(lexical / "products.parquet")
        .filter(pl.col("product_id").is_in([pid for _, pid in keys]))
        .select("product_locale", "product_id", "product_title")
        .collect(engine="streaming")
    )
    titles = {(loc, pid): title for loc, pid, title in products.iter_rows()}
    if (
        len(titles) != products.height
        or not keys.issubset(titles)
        or any(v is None for v in titles.values())
    ):
        raise ValueError("Missing or duplicate canonical product titles")
    examples = []
    for qid in ids:
        frame = selected.filter(pl.col("query_id") == qid)
        groups = frame["group_sha256"].unique().to_list()
        if len(groups) != 1 or groups[0] != found[qid]["group_sha256"]:
            raise ValueError("Query/group identity differs")
        labels = {
            (loc, pid): label
            for loc, pid, label in frame.select(
                "product_locale", "product_id", "esci_label"
            ).iter_rows()
        }
        rankings = {}
        for method in ["bm25", "hybrid"]:
            rankings[method] = [
                {
                    "rank": rank,
                    "product_locale": key[0],
                    "product_id": key[1],
                    "title": titles[tuple(key)],
                    "score": score,
                    "judgment": labels.get(tuple(key), "unknown"),
                }
                for rank, (key, score) in enumerate(found[qid]["B_scores"][method][:10], 1)
            ]
        examples.append(
            {
                "query_id": qid,
                "query": frame["query"][0],
                "group_sha256": groups[0],
                "rankings": rankings,
            }
        )
    lexical_id, dense_id = provenance["lexical_artifact_id"], provenance["dense_artifact_id"]
    payload = {
        "schema_version": 1,
        "partition": "development",
        "track": "B",
        "mode": "saved_development",
        "evaluation_contract": "v0.3",
        "retriever_k": 100,
        "rrf_constant": 60,
        "sources": {
            "report_sha256": pins["report_sha256"],
            "per_query_sha256": pins["per_query_sha256"],
            "title_projection_sha256": manifest["files"]["products.parquet"],
            "development_sha256": pins["development_sha256"],
            "evaluation_contract_sha256": pins["evaluation_contract_sha256"],
            "provenance": provenance,
        },
        "pipelines": {
            "bm25": {
                "pipeline_id": "bm25-title-lucene-v1",
                "bundle_id": lexical_id,
                "score_kind": "bm25",
                "historical_search_ms": None,
            },
            "hybrid": {
                "pipeline_id": "rrf60-m4-pretrained-v1",
                "bundle_id": f"rrf60:{lexical_id}:{dense_id}",
                "score_kind": "rrf60",
                "historical_search_ms": None,
            },
        },
        "examples": examples,
    }
    fixture = Comparison.model_validate({**payload, "fixture_id": digest(payload)})
    if len(fixture.model_dump_json().encode()) > MAX_FIXTURE_BYTES:
        raise ValueError("Demo fixture exceeds 1 MiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_once(output, fixture.model_dump())
    return {
        "path": str(output),
        "sha256": file_hash(output),
        "bytes": output.stat().st_size,
        "fixture_id": fixture.fixture_id,
        "query_ids": ids,
    }
