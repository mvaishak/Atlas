"""Separate-budget local return, native FAISS and development-only evaluation worker."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import psutil

from atlasrank.artifacts.dense import write_once
from atlasrank.data.common import check_disk
from atlasrank.retrieval.colab_environment import ROOT
from atlasrank.retrieval.colab_full import verify_coverage
from atlasrank.retrieval.colab_full_contract import approve_full
from atlasrank.retrieval.colab_full_transfer import restore_return_job
from atlasrank.retrieval.colab_pilot import restore_job
from atlasrank.retrieval.colab_runner import size
from atlasrank.retrieval.comparison import evaluate_development, verify_bm25_reference
from atlasrank.retrieval.dense_bundle import build_bundle, load_bundle
from atlasrank.retrieval.encoder import PinnedEncoder, configuration


def import_completed(
    store: Path,
    inputs: Path,
    output: Path,
    ledger: Path,
    checkpoint: Any = None,
    *,
    pilot_reference: Path | None = None,
    pilot_manifest_sha: str = "",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve earlier local imports")
    output.mkdir(parents=True)
    staging = output / "staging"
    staging.mkdir()
    job = output / "job"
    if store.is_file():
        restore_return_job(store, job, checkpoint)
    else:
        restore_job(store, job, staging, 4 * 1024**3, checkpoint)
    contract = json.loads((inputs / "CATALOG.json").read_text())
    recipe = json.loads((inputs / "REFERENCE_RECIPE.json").read_text())
    origin = json.loads((inputs / "ORIGIN.json").read_text())
    coverage = verify_coverage(
        job, inputs / "products.parquet", contract, recipe, origin, checkpoint
    )
    numerical = (
        compare_saved_pilot(job, pilot_reference, pilot_manifest_sha)
        if pilot_reference is not None
        else None
    )
    build_bundle(
        job,
        output / "bundle",
        budget_root=output.parent,
        checkpoint=checkpoint,
        accounting_directory=ledger,
    )
    _, _, bundle = load_bundle(output / "bundle", recipe, expected_origin=origin)
    return {**coverage, "artifact_id": bundle["artifact_id"], "cpu_pilot_numerical": numerical}


def worker(
    manifest: Path, inputs: Path, store: Path, work: Path, output: Path, return_sha256: str
) -> None:
    spec = approve_full(manifest, inputs, store, local=True)
    parent = int(os.environ.get("ATLAS_MAC_PARENT_PID", "-1"))
    deadline = float(os.environ.get("ATLAS_MAC_DEADLINE", "0"))
    if parent != os.getppid() or time.monotonic() >= deadline:
        raise PermissionError("Local worker requires approved supervisor")

    def guard() -> None:
        if os.getppid() != parent or time.monotonic() >= deadline:
            raise RuntimeError("Local supervisor/deadline lost")
        if (
            psutil.Process().memory_info().rss > 8 * 1024**3
            or psutil.virtual_memory().available < 1024**3
        ):
            raise RuntimeError("Local memory pressure")
        check_disk(Path("data/esci"))
        if size(work) + store.stat().st_size > spec["local_new_artifact_limit_bytes"]:
            raise RuntimeError("Local incremental reservation exceeded")

    local_ledger = work / "local-budget"
    reference = Path(spec["bm25_reference_per_query"])
    from atlasrank.retrieval.colab_transport import sha

    if sha(reference) != spec["bm25_reference_per_query_sha256"]:
        raise ValueError("Frozen M3 output changed")
    guard()
    if sha(store) != return_sha256:
        raise ValueError("Local return archive changed")
    guard()
    started = time.perf_counter()
    result = import_completed(
        store,
        inputs,
        output,
        local_ledger,
        guard,
        pilot_reference=Path(spec["cpu_pilot_reference"]),
        pilot_manifest_sha=spec["cpu_pilot_transfer_sha256"],
    )
    imported_seconds = time.perf_counter() - started
    cfg = configuration(ROOT / "configs/dense.json")
    started = time.perf_counter()
    encoder = PinnedEncoder(cfg)
    model_load_seconds = time.perf_counter() - started
    source = {
        "catalog_sha256": spec["catalog_sha256"],
        "prepared_audit_sha256": spec["prepared_audit_sha256"],
        "development_sha256": spec["development_sha256"],
    }
    recipe = json.loads((inputs / "REFERENCE_RECIPE.json").read_text())
    job = output / "job"
    comparison = output / "development"
    started = time.perf_counter()
    evaluate_development(
        Path(cfg["prepared_path"]),
        Path(spec["lexical_reference_bundle"]),
        output / "bundle",
        encoder,
        comparison,
        source,
        job,
        checkpoint=guard,
        product_recipe=recipe,
        accounting_directory=local_ledger,
    )
    verify_bm25_reference(comparison, Path(spec["bm25_reference_per_query"]))
    guard()
    write_once(
        output / "LOCAL_COMPLETE.json",
        {
            **result,
            "import_index_seconds": imported_seconds,
            "query_model_load_seconds": model_load_seconds,
            "development_seconds": time.perf_counter() - started,
            "M3_reference_exact": True,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("manifest", "inputs", "store", "work", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--return-sha256", required=True)
    a = p.parse_args()
    worker(a.manifest, a.inputs, a.store, a.work, a.output, a.return_sha256)


def compare_saved_pilot(job: Path, reference: Path, expected_manifest_sha: str) -> dict[str, Any]:
    """Rejoin hash-selected CPU pilot identities, which are not a catalog prefix."""
    import numpy as np
    import polars as pl

    from atlasrank.artifacts.dense import product_keys
    from atlasrank.evaluation.cuda_checks import compare_cuda_vectors
    from atlasrank.retrieval.chunks import completed_chunks
    from atlasrank.retrieval.colab_transport import sha

    if sha(reference / "TRANSFER.json") != expected_manifest_sha:
        raise ValueError("CPU pilot reference manifest differs")
    transfer = json.loads((reference / "TRANSFER.json").read_text())
    for name in ("pilot/products.parquet", "pilot/cpu-vectors.npy"):
        if sha(reference / name) != transfer["files"][name]:
            raise ValueError("Saved CPU pilot inputs differ")
    products = pl.read_parquet(reference / "pilot/products.parquet")
    keys = product_keys(products)
    positions = dict(zip(keys, range(len(keys)), strict=True))
    cpu = np.load(reference / "pilot/cpu-vectors.npy", allow_pickle=False)
    actual = np.zeros_like(cpu)
    found: set[int] = set()
    for _, frame, vectors in completed_chunks(job):
        for row, key in enumerate(product_keys(frame)):
            if key in positions:
                i = positions[key]
                if i in found or frame["product_title"][row] != products["product_title"][i]:
                    raise ValueError("CPU pilot identity/title alignment differs")
                actual[i] = vectors[row]
                found.add(i)
    if len(found) != len(keys):
        raise ValueError("CPU pilot identities missing from full return")
    return compare_cuda_vectors(cpu, actual)


if __name__ == "__main__":
    main()
