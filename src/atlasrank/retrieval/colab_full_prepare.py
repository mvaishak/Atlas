"""Local-only label-free full export. Never initializes an encoder or opens judgments."""

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from atlasrank.artifacts.dense import file_hash, write_once
from atlasrank.data.common import check_disk, check_memory
from atlasrank.retrieval.chunks import canonical_catalog
from atlasrank.retrieval.colab_environment import ROOT
from atlasrank.retrieval.colab_full import catalog_contract
from atlasrank.retrieval.colab_full_contract import AUDIT_SHA, CATALOG_SHA, ROWS
from atlasrank.retrieval.encoder import configuration, verify_snapshot


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve previous full exports")
    config = configuration(ROOT / "configs/dense.json")
    source = Path(config["prepared_path"]) / "catalog.parquet"
    if file_hash(source) != CATALOG_SHA or file_hash(source.parent / "audit.json") != AUDIT_SHA:
        raise ValueError("Approved source catalog/audit differs")
    check_disk(output.parent, 512 * 1024**2)
    model_hashes = verify_snapshot(config)
    output.mkdir(parents=True)
    started = time.perf_counter()
    canonical_catalog(source, output / "products.parquet")
    check_memory()
    contract = catalog_contract(output / "products.parquet", ROWS)
    contract["source_catalog_sha256"] = CATALOG_SHA
    write_once(output / "CATALOG.json", contract)
    write_once(
        output / "ORIGIN.json", {"catalog_sha256": CATALOG_SHA, "prepared_audit_sha256": AUDIT_SHA}
    )
    pilot = ROOT / "data/esci/infrastructure/m4-colab-return-v1/gpu-evidence/report.json"
    report = json.loads(pilot.read_text())
    if report["newly_encoded_titles"] != 4096:
        raise ValueError("Accepted pilot evidence missing")
    write_once(output / "REFERENCE_RECIPE.json", report["recipe"])
    for name in model_hashes:
        target = output / "model" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(config["model_path"]) / name, target)
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "src/atlasrank/py.typed",
        *sorted((ROOT / "environments/colab").iterdir()),
        ROOT / "configs/dense.json",
        ROOT / "configs/dense-cuda.json",
        ROOT / "docs/EVALUATION.md",
    ]
    for p in paths:
        target = output / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    files = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    total = sum((output / n).stat().st_size for n in files)
    if total > 512 * 1024**2:
        raise RuntimeError("Full upload limit exceeded; preserve export")
    transfer = {
        "schema_version": 1,
        "kind": "m4_full_label_free_inputs",
        "rows": ROWS,
        "files": files,
        "model_files_sha256": model_hashes,
        "expanded_bytes": total,
        "pilot_report_sha256": file_hash(pilot),
    }
    write_once(output / "FULL_TRANSFER.json", transfer)
    archive = output.parent / (output.name + ".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                z.write(p, str(p.relative_to(output)))
    check_disk(output.parent)
    result = {
        "directory": str(output),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": file_hash(archive),
        "transfer_manifest_sha256": file_hash(output / "FULL_TRANSFER.json"),
        "canonical_projection_sha256": contract["catalog_sha256"],
        "rows": ROWS,
        "chunks": len(contract["chunks"]),
        "last_chunk": contract["chunks"][-1],
        "expanded_bytes": total,
        "preparation_seconds": time.perf_counter() - started,
        "model_inference": False,
        "labels_loaded": False,
        "uploaded": False,
    }
    write_once(output.parent / (output.name + ".preparation.json"), result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
