"""Read-only M6 byte inventory. Never imports Torch/FAISS/XGBoost or authorizes a fit."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_BUNDLE = (
    "data/esci/infrastructure/m4-full-local-v1/attempt-bd83a1bac3b54cdab48ca32e79f0bf90/bundle"
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_file(path: Path, expected: str) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "expected_sha256": expected}
    if path.is_symlink():
        return {**result, "status": "symlink_rejected"}
    if not path.is_file():
        return {**result, "status": "missing"}
    actual = sha256(path)
    return {
        **result,
        "sha256": actual,
        "bytes": path.stat().st_size,
        "status": "passed" if actual == expected else "checksum_mismatch",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bundle", type=Path, default=Path(DEFAULT_BUNDLE))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence directories")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    root = args.root.resolve()
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def load(path: Path) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(path.read_text())
        return value

    def check(path: Path, expected: str) -> None:
        checks.append(verify_file(path, expected))

    def bundle(directory: Path, id_key: str, reference: dict[str, Any]) -> dict[str, Any]:
        manifest = load(directory / "manifest.json")
        complete = load(directory / "COMPLETE.json")
        check(directory / "manifest.json", complete["manifest_sha256"])
        payload = {k: v for k, v in manifest.items() if k != id_key}
        identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if identity != manifest[id_key] or manifest[id_key] != reference[id_key]:
            errors.append(f"Bundle identity mismatch: {directory}")
        for name, expected in reference["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe manifest member")
            check(directory / name, expected)
        return manifest

    try:
        review = load(root / "docs/reviews/M4.cpu-run.validation.json")
        prepared = root / "data/esci/prepared-v4"
        check(prepared / "audit.json", review["source_and_contract"]["audit_sha256"])
        audit = load(prepared / "audit.json")
        # Explicit allowlist: never open raw examples or official test judgments/references.
        for name in (
            "catalog.parquet",
            "train.parquet",
            "development.parquet",
            "query_groups.parquet",
        ):
            check(prepared / name, audit["output_sha256"][name])
        check(
            root / "docs/EVALUATION.md", review["source_and_contract"]["evaluation_contract_sha256"]
        )
        reference_dir = root / "data/esci/reports/m4-cpu-eval-v1/returned/bundle"
        check(
            reference_dir / "manifest.json",
            review["return_evidence"]["published_file_hashes"]["bundle/manifest.json"],
        )
        reference = load(reference_dir / "manifest.json")
        dense_dir = args.bundle if args.bundle.is_absolute() else root / args.bundle
        dense = bundle(dense_dir, "artifact_id", reference)
        lexical_dir = root / "data/esci/indexes/m3-bm25-v2"
        lexical = load(lexical_dir / "manifest.json")
        if lexical["bundle_id"] != review["source_and_contract"]["m3_bundle_id"]:
            errors.append("Frozen M3 lexical reference differs")
        bundle(lexical_dir, "bundle_id", lexical)
        for manifest in (dense, lexical):
            if manifest["origin"]["catalog_sha256"] != audit["output_sha256"]["catalog.parquet"]:
                errors.append("Catalog origin differs")
        config = load(root / "configs/dense.json")
        model_dir = Path(config["model_path"])
        if not model_dir.is_absolute():
            model_dir = root / model_dir
        for name, expected in reference["recipe"]["model_files_sha256"].items():
            check(model_dir / name, expected)
        feature = load(root / "configs/ranking-features-v1.json")
        actual_schema_id = hashlib.sha256(
            json.dumps(feature["schema"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual_schema_id != feature["schema_id"]:
            errors.append("Feature schema digest differs")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    versions: dict[str, str | None] = {}
    for name in (
        "torch",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "safetensors",
        "xgboost-cpu",
        "numpy",
        "scipy",
        "faiss-cpu",
        "polars",
        "pyarrow",
        "bm25s",
        "fastapi",
        "httpx",
        "psutil",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_paths = [
        *sorted((root / "src/atlasrank").rglob("*.py")),
        root / "configs/ranking-features-v1.json",
        root / "configs/m6-ranker-fixed-v1.json",
        root / "pyproject.toml",
        root / "uv.lock",
        root / "scripts/check_m6_ranker_prerequisites.py",
    ]
    source_hashes = {}
    for path in source_paths:
        try:
            source_hashes[str(path.relative_to(root))] = sha256(path)
        except OSError as error:
            errors.append(f"Source unavailable: {path}: {error}")
    result = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "status": "blocked_before_training",
        "byte_inventory_passed": not errors and all(c["status"] == "passed" for c in checks),
        "checks": checks,
        "errors": errors,
        "environment": {
            "platform": sys.platform,
            "python": sys.version,
            "machine": platform.machine(),
            "packages": versions,
        },
        "source_sha256": source_hashes,
        "executed_runner_sha256": sha256(Path(__file__).resolve()),
        "wall_seconds": time.monotonic() - start,
        "native_execution_started": False,
        "fit_executed": False,
        "training_authorized_by_this_report": False,
        "unrun_gates": [
            "compatible real-encoder Linux runtime validation and resource preflight",
            "semantic catalog/product/vector/native-index alignment",
            "normalized train/development query-group isolation",
            "real-encoder offline/HTTP post-DMatrix exact parity and edge cases",
            "objective-update sanity, full fit, A/C evaluation and fitted-model parity",
        ],
    }
    with (args.output / "validation.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "byte_inventory_passed", "errors")}))
    # Inventory success cannot satisfy the real-data training gates.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
