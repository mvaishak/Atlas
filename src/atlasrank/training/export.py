"""Build a minimal, reproducible training-only Colab input archive."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from atlasrank.data.common import sha256
from atlasrank.data.splits import normalize_query
from atlasrank.data.supervision import admissible, validate_pairs
from atlasrank.retrieval.encoder import MODEL_FILES, REPO, REVISION, verify_snapshot
from atlasrank.training.dual_encoder import (
    GroupJudgments,
    PlannedGroup,
    SamplingPlan,
    build_sampling_plan,
)
from atlasrank.training.tokenization import token_id_fingerprint

TRAIN_COLUMNS = {
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "normalized_query",
    "group_key",
    "group_sha256",
    "partition",
}
MODEL_DIRECTORY = Path("data/esci/models") / f"all-MiniLM-L6-v2-{REVISION}"


def _training_groups(path: Path, seed: int = 42) -> tuple[list[GroupJudgments], dict[str, int]]:
    frame = pl.read_parquet(path)
    if not TRAIN_COLUMNS.issubset(frame.columns):
        raise ValueError("Training manifest is missing required columns")
    if set(frame.get_column("partition").unique().to_list()) != {"train"}:
        raise ValueError("Only the train partition can be exported for M5 fitting")
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    label_counts: dict[str, int] = defaultdict(int)
    for row in frame.select(sorted(TRAIN_COLUMNS)).iter_rows(named=True):
        label = row["esci_label"]
        label_counts[str(label)] += 1
        key = (str(row["product_locale"]), str(row["group_key"]))
        group = grouped.setdefault(
            key,
            {
                "group_sha256": str(row["group_sha256"]),
                "query": str(row["query"]),
                "normalized_query": str(row["normalized_query"]),
                "judgments": [],
            },
        )
        if (
            group["group_sha256"] != row["group_sha256"]
            or group["query"] != row["query"]
            or group["normalized_query"] != row["normalized_query"]
        ):
            raise ValueError("Training group has inconsistent query identity/text")
        group["judgments"].append((str(row["product_id"]), str(label)))
    result: list[GroupJudgments] = []
    for (locale, group_key), value in grouped.items():
        judgments = validate_pairs(value["judgments"])
        if normalize_query(value["query"]) != value["normalized_query"]:
            raise ValueError("Training query normalization differs from the frozen split manifest")
        result.append(
            GroupJudgments(
                locale=locale,
                group_key=group_key,
                group_sha256=value["group_sha256"],
                query=value["query"],
                positives=tuple(sorted(admissible("train", judgments).positives)),
                negatives=tuple(sorted(admissible("train", judgments).negatives)),
            )
        )
    return result, dict(sorted(label_counts.items()))


def _selected_rows(plan: SamplingPlan, catalog: Path) -> dict[tuple[str, str], str]:
    requested = sorted(
        {
            (group.locale, product_id)
            for group in plan.groups
            for product_id in (*group.positives, *group.negatives)
        }
    )
    key_frame = pl.DataFrame(requested, schema=["product_locale", "product_id"], orient="row")
    selected = (
        pl.scan_parquet(catalog)
        .select("product_locale", "product_id", "product_title")
        .join(key_frame.lazy(), on=["product_locale", "product_id"], how="inner")
        .collect()
        .sort(["product_locale", "product_id"])
    )
    if selected.height != len(requested):
        raise ValueError("Some selected training judgments do not join to the canonical catalog")
    if selected.select(["product_locale", "product_id"]).n_unique() != len(requested):
        raise ValueError("Selected catalog identities are duplicated")
    return {
        (str(row["product_locale"]), str(row["product_id"])): str(row["product_title"] or "")
        for row in selected.iter_rows(named=True)
    }


def _group_record(group: PlannedGroup, titles: dict[tuple[str, str], str]) -> dict[str, Any]:
    key = group.locale
    return {
        "locale": group.locale,
        "group_key": group.group_key,
        "group_sha256": group.group_sha256,
        "query": group.query,
        "positives": [
            {"product_id": product_id, "title": titles[(key, product_id)]}
            for product_id in group.positives
        ],
        "negatives": [
            {"product_id": product_id, "title": titles[(key, product_id)]}
            for product_id in group.negatives
        ],
    }


def _zip_write(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def build_training_export(
    repository: Path,
    output: Path,
    *,
    seed: int = 42,
    max_positives: int = 2,
    max_negatives: int = 4,
) -> dict[str, Any]:
    prepared = repository / "data/esci/prepared-v4"
    train_path = prepared / "train.parquet"
    catalog_path = prepared / "catalog.parquet"
    model_root = repository / MODEL_DIRECTORY
    dense_config: dict[str, Any] = json.loads((repository / "configs/dense.json").read_text())
    dense_config["model_path"] = str(model_root)
    groups, label_counts = _training_groups(train_path, seed)
    plan = build_sampling_plan(
        groups, seed=seed, max_positives=max_positives, max_negatives=max_negatives
    )
    titles = _selected_rows(plan, catalog_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root), local_files_only=True, trust_remote_code=False
    )
    unique_texts = [group.query for group in plan.groups]
    unique_texts.extend(titles.values())
    token_fingerprint = token_id_fingerprint(tokenizer, unique_texts)
    model_hashes: dict[str, str] = {}
    verify_snapshot(dense_config)
    if sha256(catalog_path) != dense_config["catalog_sha256"]:
        raise ValueError("Current catalog differs from the accepted M4 source pin")
    for relative in sorted(MODEL_FILES):
        target = model_root / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"Pinned model file is missing or unsafe: {relative}")
        model_hashes[relative] = sha256(target)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Training exports are immutable; choose a new output path")
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".m5-export-", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            _zip_write(archive, "sampling-plan.json", plan.canonical_bytes())
            records = b"".join(
                (
                    json.dumps(_group_record(group, titles), ensure_ascii=False, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
                for group in plan.groups
            )
            _zip_write(archive, "train-groups.jsonl", records)
            for relative in sorted(MODEL_FILES):
                _zip_write(archive, f"model/{relative}", (model_root / relative).read_bytes())
            source_paths = [
                repository / "src/atlasrank/data/splits.py",
                repository / "src/atlasrank/data/supervision.py",
                repository / "src/atlasrank/data/common.py",
                repository / "src/atlasrank/training/__init__.py",
                repository / "src/atlasrank/training/dual_encoder.py",
                repository / "src/atlasrank/training/model.py",
                repository / "src/atlasrank/training/tokenization.py",
                repository / "src/atlasrank/training/checkpoints.py",
                repository / "src/atlasrank/training/loop.py",
                repository / "src/atlasrank/training/budget.py",
                repository / "src/atlasrank/training/runner.py",
                repository / "src/atlasrank/training/configuration.py",
                repository / "src/atlasrank/training/observations.py",
                repository / "src/atlasrank/training/export.py",
                repository / "scripts/run_m5_training.py",
                repository / "src/atlasrank/retrieval/encoder.py",
                repository / "src/atlasrank/retrieval/colab_runner.py",
                repository / "src/atlasrank/retrieval/colab_environment.py",
                repository / "src/atlasrank/retrieval/colab_transport.py",
            ]
            source_hashes = {
                str(path.relative_to(repository)): sha256(path) for path in source_paths
            }
            for path in source_paths:
                _zip_write(archive, f"source/{path.relative_to(repository)}", path.read_bytes())
            for environment_relative in (
                Path("environments/colab/environment.json"),
                Path("environments/colab/requirements.txt"),
            ):
                path = repository / environment_relative
                source_hashes[str(environment_relative)] = sha256(path)
                _zip_write(archive, str(environment_relative), path.read_bytes())
            train_hash = sha256(train_path)
            catalog_hash = sha256(catalog_path)
            audit_hash = sha256(prepared / "audit.json")
            if audit_hash != dense_config["prepared_audit_sha256"]:
                raise ValueError("Prepared-data audit differs from the accepted M4 source pin")
            evaluation_contract_hash = sha256(repository / "docs/EVALUATION.md")
            dense_config_hash = sha256(repository / "configs/dense.json")
            m4_validation_path = repository / "docs/reviews/M4.cpu-run.validation.json"
            m4_validation: dict[str, Any] = json.loads(m4_validation_path.read_text())
            if (
                m4_validation["completion"]["status"] != "passed"
                or not m4_validation["completion"]["m3_reference_exact"]
            ):
                raise ValueError("Accepted M4 baseline verification is missing")
            m4_reference = {
                "m3_bm25_bundle_id": m4_validation["source_and_contract"]["m3_bundle_id"],
                "m3_reference_sha256": m4_validation["source_and_contract"]["m3_reference_sha256"],
                "m4_dense_bundle_id": m4_validation["completion"]["artifact_id"],
                "m4_report_archive_sha256": sha256(
                    repository / "data/esci/reports/m4-cpu-eval-v1/colab-results.zip"
                ),
                "rrf_k": 60,
            }
            export_metadata = {
                "schema_version": 1,
                "kind": "atlasrank_m5_training_only",
                "partition": "train",
                "train_manifest_sha256": train_hash,
                "train_rows": sum(label_counts.values()),
                "catalog_sha256": catalog_hash,
                "prepared_audit_sha256": audit_hash,
                "dense_cpu_config_sha256": dense_config_hash,
                "exporter_sha256": sha256(repository / "scripts/prepare_m5_export.py"),
                "evaluation_contract_sha256": evaluation_contract_hash,
                "m4_reference": m4_reference,
                "m4_validation_sha256": sha256(m4_validation_path),
                "token_id_fingerprint": token_fingerprint,
                "sampling_plan_sha256": plan.sha256,
                "training_groups_sha256": hashlib.sha256(records).hexdigest(),
                "training_groups": plan.usable_group_count,
                "source_groups": plan.source_group_count,
                "excluded_groups": plan.excluded_group_count,
                "globally_masked_negative_count": plan.globally_masked_negative_count,
                "label_counts": label_counts,
                "selected_product_identity_count": len(titles),
                "selected_positive_pair_count": sum(len(group.positives) for group in plan.groups),
                "selected_negative_pair_count": sum(len(group.negatives) for group in plan.groups),
                "selected_product_title_bytes_utf8": sum(
                    len(v.encode("utf-8")) for v in titles.values()
                ),
                "sampling_seed": seed,
                "max_positives": max_positives,
                "max_negatives": max_negatives,
                "model_repo": REPO,
                "model_revision": REVISION,
                "model_file_sha256": model_hashes,
                "source_sha256": source_hashes,
                "contents_excluded": [
                    "development labels",
                    "official test judgments",
                    "full catalog",
                    "unselected product titles",
                    "raw source data",
                ],
            }
            _zip_write(
                archive,
                "EXPORT.json",
                (json.dumps(export_metadata, sort_keys=True, indent=2) + "\n").encode(),
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **export_metadata,
        "archive_path": str(output),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256(output),
        "sampling_plan_bytes": len(plan.canonical_bytes()),
        "training_groups_bytes": len(records),
    }
