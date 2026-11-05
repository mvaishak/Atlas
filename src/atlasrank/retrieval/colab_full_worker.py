"""Streaming full-catalog CUDA worker. No labels, FAISS, training or tuning."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import psutil

from atlasrank.artifacts.dense import digest, write_once
from atlasrank.retrieval.colab_environment import ROOT, validate_installed
from atlasrank.retrieval.colab_full import catalog_contract, encode_persisted
from atlasrank.retrieval.colab_full_contract import approve_full
from atlasrank.retrieval.colab_runner import size
from atlasrank.retrieval.colab_transport import publish, sha
from atlasrank.retrieval.cuda_encoder import CudaEncoder
from atlasrank.retrieval.encoder import configuration


def worker(manifest: Path, inputs: Path, persistent: Path, work: Path) -> dict[str, Any]:
    spec = approve_full(manifest, inputs, persistent)
    declared = json.loads((inputs / "CATALOG.json").read_text())
    actual = catalog_contract(inputs / "products.parquet", spec["catalog_products"])
    if any(declared.get(k) != v for k, v in actual.items()):
        raise ValueError("Actual full row coverage differs from approved export")
    parent = int(os.environ.get("ATLAS_COLAB_PARENT_PID", "-1"))
    deadline = float(os.environ.get("ATLAS_COLAB_DEADLINE", "0"))
    if parent != os.getppid() or time.monotonic() >= deadline:
        raise PermissionError("Full worker requires active approved supervisor")
    validate_installed(json.loads((ROOT / "environments/colab/environment.json").read_text()))
    import torch

    if (
        torch.__file__ != os.environ["ATLAS_COLAB_NATIVE_TORCH_FILE"]
        or str(torch.__version__) != os.environ["ATLAS_COLAB_NATIVE_TORCH_VERSION"]
    ):
        raise RuntimeError("Native Torch replaced")
    if psutil.virtual_memory().available < spec["available_ram_start_bytes"]:
        raise RuntimeError("Insufficient starting host RAM")
    cpu = configuration(ROOT / "configs/dense.json")
    cpu["model_path"] = str(inputs / "model")
    cuda = json.loads((ROOT / "configs/dense-cuda.json").read_text())
    started = time.perf_counter()
    encoder = CudaEncoder(cpu, cuda)
    load_seconds = time.perf_counter() - started
    reference = json.loads((inputs / "REFERENCE_RECIPE.json").read_text())
    if encoder.recipe != reference or digest(reference) != spec["cuda_recipe_id"]:
        raise ValueError("Full CUDA recipe differs from accepted pilot; stop for review")

    next_storage_check = 0.0

    def guard() -> None:
        nonlocal next_storage_check
        if os.getppid() != parent or time.monotonic() >= deadline:
            raise RuntimeError("Full supervisor/deadline lost")
        if (
            psutil.Process().memory_info().rss > spec["rss_limit_bytes"]
            or psutil.virtual_memory().available < 1024**3
        ):
            raise RuntimeError("Full host memory limit")
        if time.monotonic() >= next_storage_check and (
            size(persistent) > spec["persistent_limit_bytes"]
            or size(work.parent) + spec["scratch_input_reservation_bytes"]
            > spec["scratch_limit_bytes"]
        ):
            raise RuntimeError("Full storage cap")
        if time.monotonic() >= next_storage_check:
            next_storage_check = time.monotonic() + 5
        encoder.check_memory()

    (persistent / "progress").mkdir(exist_ok=True)
    evidence = work / "evidence"
    evidence.mkdir()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    events = evidence / "progress.jsonl"
    with events.open("x") as log:

        def event(record: dict[str, Any]) -> None:
            stage = str(record["stage"])
            counts[stage] = counts.get(stage, 0) + 1
            totals[stage] = totals.get(stage, 0) + float(record.get("seconds", 0))
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if stage == "checkpoint_publication":
                # Small immutable progress record survives a killed worker/runtime.
                from atlasrank.retrieval.colab_transport import once

                once(
                    persistent / "progress" / (uuid.uuid4().hex + ".json"),
                    {
                        "completed_chunk_start": record["chunk_start"],
                        "stage_seconds": dict(totals),
                        "stage_counts": dict(counts),
                        "batches": len(encoder.timings),
                        "encoder_seconds": {
                            k: sum(float(t[k]) for t in encoder.timings)
                            for k in (
                                "tokenization_seconds",
                                "model_inference_seconds",
                                "host_to_device_seconds",
                                "device_to_host_seconds",
                            )
                        },
                    },
                )
            if stage in ("checkpoint_publication", "chunk_reused", "coverage_verified"):
                print(json.dumps(record), flush=True)

        guard()
        started = time.perf_counter()
        try:
            result = encode_persisted(
                inputs / "products.parquet",
                work,
                encoder,
                json.loads((inputs / "ORIGIN.json").read_text()),
                persistent,
                chunk_rows=1024,
                wall_seconds=18000,
                checkpoint=guard,
                event=event,
            )
        except BaseException:
            # Preserve whatever log is complete; budget/supervisor may terminate cleanup.
            publish(evidence, persistent / "events", uuid.uuid4().hex, 4 * 1024**3)
            raise
        elapsed = time.perf_counter() - started
    stages = {
        name: sum(float(t[name]) for t in encoder.timings)
        for name in (
            "tokenization_seconds",
            "host_to_device_seconds",
            "model_inference_seconds",
            "device_to_host_seconds",
            "end_to_end_seconds",
        )
    }
    report = {
        **result,
        "schema_version": 1,
        "scope": "full catalog encoding only",
        "recipe": encoder.recipe,
        "pipeline_seconds": elapsed,
        "model_load_seconds": load_seconds,
        "batch_stage_seconds": stages,
        "stage_seconds": totals,
        "stage_counts": counts,
        "newly_encoded_titles": sum(int(t["titles"]) for t in encoder.timings),
        "batch_timings": encoder.timings,
        "memory": encoder.check_memory(),
        "labels_loaded": False,
        "training": False,
        "test_judgments_accessed": False,
    }
    write_once(evidence / "report.json", report)
    write_once(evidence / "COMPLETE.json", {"report_sha256": sha(evidence / "report.json")})
    guard()
    publish(evidence, persistent / "evidence", uuid.uuid4().hex, 4 * 1024**3)
    guard()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "inputs", "persistent", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    worker(a.manifest, a.inputs, a.persistent, a.work)


if __name__ == "__main__":
    main()
