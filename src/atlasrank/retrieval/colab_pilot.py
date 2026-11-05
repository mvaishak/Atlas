"""Bounded CUDA worker: existing immutable chunk engine with verified persistence."""

import argparse
import importlib.metadata
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil

from atlasrank.artifacts.dense import file_hash, product_keys, write_once
from atlasrank.evaluation.cuda_checks import compare_cuda_vectors
from atlasrank.retrieval.chunks import completed_chunks, encode_chunks, validate_chunk
from atlasrank.retrieval.colab_environment import ROOT, approval, validate_installed
from atlasrank.retrieval.colab_runner import size
from atlasrank.retrieval.colab_transport import copy_verified, publish, restore, sha
from atlasrank.retrieval.cuda_encoder import CudaEncoder, token_arrays, token_hashes
from atlasrank.retrieval.encoder import configuration


def persist_metadata(job: Path, store: Path, staging: Path, limit: int) -> None:
    paths = [p for p in job.glob("*.json") if p.name in {"JOB.json", "JOB_COMPLETE.json"}]
    paths += list((job / "attempts").glob("*.json"))
    for source in paths:
        relative = str(source.relative_to(job))
        single = staging / uuid.uuid4().hex
        (single / relative).parent.mkdir(parents=True, exist_ok=True)
        copy_verified(source, single / relative, sha(source))
        import hashlib

        publish(single, store / "control", hashlib.sha256(relative.encode()).hexdigest(), limit)


def restore_job(
    store: Path, job: Path, staging: Path, limit: int, checkpoint: Callable[[], None] | None = None
) -> None:
    job.mkdir()
    for receipt in sorted((store / "control").glob("*.receipt.json")):
        if checkpoint:
            checkpoint()
        temporary = staging / uuid.uuid4().hex
        restore(store / "control", receipt.name.removesuffix(".receipt.json"), temporary, limit)
        for source in temporary.rglob("*"):
            if not source.is_file():
                continue
            rel = source.relative_to(temporary)
            if str(rel) not in {"JOB.json", "JOB_COMPLETE.json"} and not (
                len(rel.parts) == 2 and rel.parts[0] == "attempts" and rel.suffix == ".json"
            ):
                raise ValueError("Unexpected restored job control file")
            target = job / rel
            target.parent.mkdir(exist_ok=True)
            copy_verified(source, target, sha(source))
    (job / "chunks").mkdir()
    if (job / "JOB.json").exists():
        spec = json.loads((job / "JOB.json").read_text())
        for receipt in sorted((store / "chunks").glob("*.receipt.json")):
            if checkpoint:
                checkpoint()
            name = receipt.name.removesuffix(".receipt.json")
            if len(name) != 12 or not name.isdigit():
                raise ValueError("Invalid persistent chunk interval")
            target = job / "chunks" / name
            restore(store / "chunks", name, target, limit)
            validate_chunk(target, spec)
            for f in target.iterdir():
                f.chmod(0o444)
    elif list((store / "chunks").glob("*.receipt.json")):
        raise ValueError("Persistent chunks missing job lineage")


def pilot(manifest: Path, inputs: Path, persistent: Path, work: Path) -> dict[str, Any]:
    spec = approval(manifest, inputs, persistent)
    parent = int(os.environ.get("ATLAS_COLAB_PARENT_PID", "-1"))
    deadline = float(os.environ.get("ATLAS_COLAB_DEADLINE", "0"))
    if parent != os.getppid() or time.monotonic() >= deadline:
        raise PermissionError("CUDA worker requires its active approved supervisor")
    environment = json.loads((ROOT / "environments/colab/environment.json").read_text())
    versions = validate_installed(environment)
    import torch

    if torch.__file__ != os.environ.get("ATLAS_COLAB_NATIVE_TORCH_FILE") or str(
        torch.__version__
    ) != os.environ.get("ATLAS_COLAB_NATIVE_TORCH_VERSION"):
        raise RuntimeError("Overlay replaced native Torch")
    products = pl.read_parquet(inputs / "pilot/products.parquet")
    keys = product_keys(products)
    if len(keys) != spec["titles"]:
        raise ValueError("Pilot count differs")
    if psutil.virtual_memory().available < spec["available_ram_start_bytes"]:
        raise RuntimeError("Pilot requires 4 GiB available host RAM at model load")
    cpu_spec = configuration(ROOT / "configs/dense.json")
    cpu_spec["model_path"] = str(inputs / "model")
    cuda_recipe = json.loads((ROOT / "configs/dense-cuda.json").read_text())
    for name in ("device", "gpu_name", "batch_size", "vram_limit_bytes", "tf32", "autocast"):
        if cuda_recipe[name] != spec[name]:
            raise ValueError("CUDA recipe differs from approved pilot")
    for name in (
        "max_length",
        "dimensions",
        "dtype",
        "attention",
        "preprocessing",
        "seed",
        "threads",
    ):
        if cuda_recipe[name] != cpu_spec[name]:
            raise ValueError("CUDA recipe changes frozen CPU model semantics")
    load_started = time.perf_counter()
    encoder = CudaEncoder(cpu_spec, cuda_recipe)
    model_load_seconds = time.perf_counter() - load_started
    model_load_memory = encoder.check_memory()
    expected = json.loads((inputs / "pilot/token-hashes.json").read_text())
    titles = products["product_title"].to_list()
    actual = token_hashes(token_arrays(encoder.model.tokenizer, titles))
    if actual != expected:
        raise ValueError("Transferred CPU token IDs/masks/types differ from CUDA tokenizer")

    def guard() -> None:
        if os.getppid() != parent or time.monotonic() >= deadline:
            raise RuntimeError("Supervisor lost or pilot deadline reached")
        if (
            psutil.Process().memory_info().rss > spec["rss_limit_bytes"]
            or psutil.virtual_memory().available < 1024**3
        ):
            raise RuntimeError("Pilot RAM pressure")
        if (
            size(persistent) > spec["persistent_limit_bytes"]
            or size(work.parent) > spec["scratch_limit_bytes"]
        ):
            raise RuntimeError("Pilot storage pressure")
        encoder.check_memory()

    warm_started = time.perf_counter()
    for i in range(spec["warmup_batches"]):
        guard()
        encoder.encode(titles[i * 32 : (i + 1) * 32])
    warmup_seconds = time.perf_counter() - warm_started
    warmup_memory = encoder.check_memory()
    encoder.timings.clear()
    torch.cuda.reset_peak_memory_stats(0)
    staging = work / "staging"
    staging.mkdir()
    job = work / "job"
    restore_job(persistent, job, staging, spec["persistent_limit_bytes"])
    checkpoint_seconds = 0.0

    def commit(chunk: Path) -> None:
        nonlocal checkpoint_seconds
        guard()
        started = time.perf_counter()
        persist_metadata(job, persistent, staging, spec["persistent_limit_bytes"])
        publish(chunk, persistent / "chunks", chunk.name, spec["persistent_limit_bytes"])
        checkpoint_seconds += time.perf_counter() - started
        guard()

    started = time.perf_counter()
    try:
        result = encode_chunks(
            inputs / "pilot/products.parquet",
            job,
            encoder,
            json.loads((inputs / "pilot/origin.json").read_text()),
            chunk_rows=1024,
            wall_seconds=600,
            budget_root=work,
            checkpoint=guard,
            on_chunk=commit,
        )
    finally:
        persist_metadata(job, persistent, staging, spec["persistent_limit_bytes"])
    measured_seconds = time.perf_counter() - started
    rows = list(completed_chunks(job))
    cuda = np.concatenate([v for _, _, v in rows])
    cpu = np.load(inputs / "pilot/cpu-vectors.npy", allow_pickle=False)
    numerical = compare_cuda_vectors(cpu, cuda)
    guard()
    single = encoder.encode([titles[0]])
    np.testing.assert_array_equal(single, encoder.encode([titles[0]]))
    np.testing.assert_allclose(single, cuda[:1], rtol=1e-4, atol=1e-5)
    report_dir = work / "evidence"
    report_dir.mkdir()
    report = {
        "schema_version": 1,
        "scope": "CUDA pilot, no relevance judgments/training/full catalog",
        "recipe": encoder.recipe,
        "job_id": result["job_id"],
        "tokens_exact": True,
        "ordered_product_keys_sha256": file_hash(inputs / "pilot/products.parquet"),
        "numerical": numerical,
        "same_device_single_repeat_exact": True,
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "setup_seconds": float(os.environ["ATLAS_COLAB_SETUP_SECONDS"]),
        "measured_pipeline_seconds": measured_seconds,
        "newly_encoded_titles": sum(t["titles"] for t in encoder.timings[:-2]),
        "end_to_end_titles_per_second": (
            sum(t["titles"] for t in encoder.timings[:-2]) / measured_seconds
        ),
        "persistent_checkpoint_seconds": checkpoint_seconds,
        "batch_timings": encoder.timings,
        "memory": encoder.check_memory(),
        "model_load_memory": model_load_memory,
        "warmup_memory": warmup_memory,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "pinned_overlay_versions": versions,
        "installed_distribution_versions": {
            name: importlib.metadata.version(name)
            for name in sorted({d.metadata["Name"] for d in importlib.metadata.distributions()})
        },
        "full_run_recommendation": "Pending local FAISS review and separate full-run approval",
    }
    write_once(report_dir / "report.json", report)
    write_once(report_dir / "COMPLETE.json", {"report_sha256": sha(report_dir / "report.json")})
    publish(report_dir, persistent / "evidence", uuid.uuid4().hex, spec["return_limit_bytes"])
    guard()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "inputs", "persistent", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    pilot(args.manifest, args.inputs, args.persistent, args.work)


if __name__ == "__main__":
    main()
