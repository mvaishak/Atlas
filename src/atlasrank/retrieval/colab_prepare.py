"""Prepare a private label-free transfer locally, without inference or Hub access."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from atlasrank.artifacts.dense import file_hash, product_keys, write_once
from atlasrank.retrieval.colab_environment import ROOT
from atlasrank.retrieval.cuda_encoder import token_arrays, token_hashes
from atlasrank.retrieval.encoder import configuration, validate_vectors, verify_snapshot


def prepare(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve previous transfer preparations")
    spec = configuration(ROOT / "configs/dense.json")
    pilot = ROOT / "data/esci/calibrations/m4-cpu-v1"
    report = json.loads((pilot / "report.json").read_text())
    for name, pin in (("sample.parquet", "sample_sha256"), ("vectors.npy", "vectors_sha256")):
        if file_hash(pilot / name) != report[pin]:
            raise ValueError("CPU calibration input/output changed")
    if report["catalog_sha256"] != spec["catalog_sha256"]:
        raise ValueError("CPU pilot differs from approved catalog")
    hashes = verify_snapshot(spec)
    if hashes != report["model_file_sha256"]:
        raise ValueError("CPU pilot model differs")
    sample = pl.read_parquet(pilot / "sample.parquet").select(
        "product_locale", "product_id", "product_title"
    )
    if len(product_keys(sample)) != 4096:
        raise ValueError("Expected exact canonical CPU sample")
    vectors = np.load(pilot / "vectors.npy", allow_pickle=False)
    validate_vectors(vectors, 384)
    if vectors.shape != (4096, 384):
        raise ValueError("CPU reference rows differ")
    output.mkdir(parents=True)
    (output / "pilot").mkdir()
    sample.write_parquet(output / "pilot/products.parquet", compression="zstd", row_group_size=1024)
    shutil.copyfile(pilot / "vectors.npy", output / "pilot/cpu-vectors.npy")
    origin = {
        "catalog_sha256": spec["catalog_sha256"],
        "prepared_audit_sha256": spec["prepared_audit_sha256"],
        "cpu_pilot_sample_sha256": report["sample_sha256"],
        "cpu_pilot_vectors_sha256": report["vectors_sha256"],
        "scope": "Exact 4096 CPU pilot identities/titles; no labels",
    }
    write_once(output / "pilot/origin.json", origin)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_path"], local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokens = token_arrays(tokenizer, sample["product_title"].to_list())
    token_payload: dict[str, Any] = dict(tokens)
    np.savez_compressed(output / "pilot/tokens.npz", **token_payload)
    write_once(output / "pilot/token-hashes.json", token_hashes(tokens))
    for name in hashes:
        target = output / "model" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(spec["model_path"]) / name, target)
    source_paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "src/atlasrank/py.typed",
        *sorted((ROOT / "tests").glob("*.py")),
        *sorted((ROOT / "environments/colab").iterdir()),
        ROOT / "configs/dense.json",
        ROOT / "configs/dense-cuda.json",
        ROOT / "docs/EVALUATION.md",
    ]
    for source in source_paths:
        target = output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Exact whitelist only. No .git, local databases, logs, judgments or raw examples.
    files = {
        str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    transfer = {
        "schema_version": 1,
        "kind": "m4_cpu_exact_label_free_pilot",
        "rows": 4096,
        "origin": origin,
        "files": files,
        "model_files_sha256": hashes,
    }
    write_once(output / "TRANSFER.json", transfer)
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if total > 256 * 1024**2:
        raise RuntimeError("Pilot upload budget exceeded; preserve preparation")
    archive = output.parent / (output.name + ".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                z.write(p, str(p.relative_to(output)))
    proposal = json.loads((ROOT / "configs/m4-colab-pilot.proposed.json").read_text())
    proposal["transfer_manifest_sha256"] = file_hash(output / "TRANSFER.json")
    write_once(output.parent / (output.name + ".pilot.proposed.json"), proposal)
    result = {
        "transfer_manifest_sha256": proposal["transfer_manifest_sha256"],
        "archive": str(archive),
        "archive_sha256": file_hash(archive),
        "archive_bytes": archive.stat().st_size,
        "expanded_bytes": total,
        "rows": 4096,
        "cpu_tokenization_only": True,
        "new_model_inference": False,
        "uploaded": False,
    }
    write_once(output.parent / (output.name + ".preparation.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
