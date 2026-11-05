"""Approved CPU execution control only; frozen AtlasRank modules perform all ML work."""

import argparse
import gc
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

GIB = 1024**3


def approved_spec(path: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text())
    if not all(
        spec.get(k) is True for k in ("execution_approved", "runnable", "destination_confirmed")
    ):
        raise PermissionError("Explicit CPU execution/destination approval required")
    frozen = {
        "kind": "m4_colab_cpu_development_evaluation",
        "device": "cpu",
        "threads": 1,
        "batch_size": 32,
        "precision": "float32",
        "catalog_products": 1215854,
        "chunks": 1188,
        "development_queries": 4129,
        "evaluation_contract": "v0.3",
        "retriever_k": 100,
        "rrf_constant": 60,
        "wall_seconds": 7200,
        "prior_cpu_preflight_reservation_seconds": 600,
        "rss_limit_bytes": 12 * GIB,
        "available_ram_start_bytes": 16 * GIB,
        "available_ram_floor_bytes": 4 * GIB,
        "scratch_limit_bytes": 16 * GIB,
        "persistent_new_limit_bytes": 8 * GIB,
        "persistent_combined_limit_bytes": 12 * GIB,
        "free_disk_reserve_bytes": 2 * GIB,
        "component_rtol": 1e-4,
        "component_atol": 1e-5,
        "cosine_atol": 1e-4,
    }
    if any(spec.get(k) != v for k, v in frozen.items()):
        raise ValueError("CPU limits/recipe differ from approved configuration")
    destination = "/content/drive/MyDrive/AtlasRank/m4/cpu-eval-v1"
    if (
        spec.get("persistent_destination") != destination
        or spec.get("proposed_persistent_destination") != destination
        or not spec.get("operator_owner_token")
    ):
        raise ValueError("CPU private destination/owner differs")
    return spec


def remaining_seconds(charged: float) -> float:
    if charged < 0:
        raise ValueError("Negative CPU charge")
    remaining = 7200 - 600 - charged
    if remaining <= 0:
        raise RuntimeError("Cumulative CPU budget exhausted")
    return remaining


def platform_guard() -> None:
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or platform.python_version() != "3.13.18"
        or os.environ.get("KMP_DUPLICATE_LIB_OK")
    ):
        raise ValueError("Only the approved Linux CPU runtime may execute")


def adapter(root: Path, spec: dict[str, Any]) -> Any:
    from atlasrank.retrieval.colab_transport import sha

    path = root / "adapter/m4_colab_cpu.py"
    if sha(path) != spec["adapter_sha256"]:
        raise ValueError("Reviewed CPU adapter changed")
    module_spec = importlib.util.spec_from_file_location("cpu_adapter", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def resources(
    spec: dict[str, Any], *, start: bool = False, publication_bytes: int = 0
) -> dict[str, int]:
    import psutil

    from atlasrank.retrieval.colab_runner import size

    available = psutil.virtual_memory().available
    threshold = spec["available_ram_start_bytes" if start else "available_ram_floor_bytes"]
    scratch = sum(
        size(p) if p.is_dir() else p.stat().st_size
        for p in Path("/content").glob("atlasrank-cpu-*")
    )
    base = Path("/content/drive/MyDrive/AtlasRank/m4")
    amounts = {n: size(base / n) for n in spec["persistent_combined_scope"]}
    added = sum(v for n, v in amounts.items() if n != "full-l4-v1")
    rss = psutil.Process().memory_info().rss
    if (
        available < threshold
        or rss > spec["rss_limit_bytes"]
        or scratch > spec["scratch_limit_bytes"]
        or added + publication_bytes > spec["persistent_new_limit_bytes"]
        or sum(amounts.values()) + publication_bytes > spec["persistent_combined_limit_bytes"]
        or shutil.disk_usage("/content").free < spec["free_disk_reserve_bytes"]
    ):
        raise RuntimeError("Approved CPU RAM/disk/storage ceiling failed; preserve all artifacts")
    return {
        "available_ram_bytes": available,
        "worker_rss_bytes": rss,
        "scratch_bytes": scratch,
        "persistent_new_bytes": added,
        "persistent_combined_bytes": sum(amounts.values()),
    }


def worker(manifest: Path, work: Path, output: Path, persistent: Path) -> None:
    spec = approved_spec(manifest)
    platform_guard()
    sys.dont_write_bytecode = True
    from atlasrank.artifacts.dense import write_once
    from atlasrank.retrieval.colab_environment import reviewed_source_hashes
    from atlasrank.retrieval.colab_full_contract import verify_inputs
    from atlasrank.retrieval.colab_runner import size
    from atlasrank.retrieval.colab_transport import copy_verified, sha

    root = Path(spec["input_root"])
    helper = adapter(root, spec)
    transfer = helper.verify_export(root, spec["transfer_manifest_sha256"])
    if reviewed_source_hashes() != spec["frozen_source_sha256"]:
        raise ValueError("Frozen package source differs")
    verify_inputs(root / "full-input", transfer["full_contract"])
    environment = json.loads((root / "environment/environment.json").read_text())
    if {n: md.version(n) for n in environment["pinned_packages"]} != environment["pinned_packages"]:
        raise ValueError("CPU overlay drift")
    parent = int(os.environ["ATLAS_MAC_PARENT_PID"])
    deadline = float(os.environ["ATLAS_MAC_DEADLINE"])
    last_check, last_progress = 0.0, 0.0
    measurements: list[dict[str, Any]] = []
    stage = "input_validation"
    output.mkdir(exist_ok=False)

    def event(name: str, **values: Any) -> None:
        record = {"event": name, "utc": time.time(), **values}
        print(json.dumps(record), flush=True)
        with (output / "progress.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def guard() -> None:
        nonlocal last_check, last_progress
        now = time.monotonic()
        if os.getppid() != parent or now >= deadline:
            raise RuntimeError("CPU supervisor/deadline lost")
        if now - last_check >= 5:
            record: dict[str, Any] = {**resources(spec), "utc": time.time()}
            measurements.append(record)
            last_check = now
        if now - last_progress >= 20:
            path = output / "development/per-query.jsonl"
            count = 0
            if path.exists():
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(1024**2), b""):
                        count += block.count(b"\n")
            event("progress", stage=stage, completed_query_records=count)
            last_progress = now

    resources(spec, start=True)
    (output / "runtime").mkdir()
    event("runtime_checked", runtime=helper.runtime_smoke(output / "runtime"))
    from atlasrank.retrieval.colab_full import verify_coverage
    from atlasrank.retrieval.colab_full_return import compare_saved_pilot
    from atlasrank.retrieval.colab_full_transfer import restore_return_job
    from atlasrank.retrieval.comparison import evaluate_development, verify_bm25_reference
    from atlasrank.retrieval.dense_bundle import build_bundle, load_bundle
    from atlasrank.retrieval.encoder import PinnedEncoder, configuration

    timings = {}
    started = time.monotonic()
    archive = work / "full-return.zip"
    if archive.exists():
        if sha(archive) != spec["return_archive_sha256"]:
            raise ValueError("Preserved local return differs")
    else:
        copy_verified(
            Path(spec["gpu_store"]) / "full-return.zip", archive, spec["return_archive_sha256"]
        )
    timings["return_copy_seconds"] = time.monotonic() - started
    guard()
    inputs = root / "full-input"
    contract = json.loads((inputs / "CATALOG.json").read_text())
    recipe = json.loads((inputs / "REFERENCE_RECIPE.json").read_text())
    origin = json.loads((inputs / "ORIGIN.json").read_text())
    job = output / "job"
    stage = "restore_and_canonical_validation"
    started = time.monotonic()
    restore_return_job(archive, job, guard)
    coverage = verify_coverage(job, inputs / "products.parquet", contract, recipe, origin, guard)
    write_once(output / "coverage.json", coverage)
    timings["restore_canonical_seconds"] = time.monotonic() - started
    stage = "saved_cpu_numerical_comparison"
    started = time.monotonic()
    numerical = compare_saved_pilot(
        job, Path(spec["cpu_reference"]), spec["cpu_pilot_transfer_sha256"]
    )
    write_once(output / "numerical.json", numerical)
    timings["saved_cpu_comparison_seconds"] = time.monotonic() - started
    event("numerical_passed", rows=4096)
    budget = work / "compute-budget"
    stage = "native_faiss_build_validation"
    started = time.monotonic()
    build_bundle(
        job, output / "bundle", budget_root=work, checkpoint=guard, accounting_directory=budget
    )
    loaded = load_bundle(output / "bundle", recipe, expected_origin=origin)
    bundle_id = loaded[2]["artifact_id"]
    del loaded
    gc.collect()
    timings["faiss_build_load_validation_seconds"] = time.monotonic() - started
    event("native_faiss_validated", rows=coverage["rows"], artifact_id=bundle_id)
    cfg = configuration(inputs / "source/configs/dense.json")
    cfg["model_path"] = str(inputs / "model")
    stage = "query_model_load"
    started = time.monotonic()
    encoder = PinnedEncoder(cfg)
    timings["query_model_load_seconds"] = time.monotonic() - started
    stage = "development_evaluation"
    started = time.monotonic()
    source = {k: spec[k] for k in ("catalog_sha256", "prepared_audit_sha256", "development_sha256")}
    evaluate_development(
        Path(spec["prepared_path"]),
        Path(spec["frozen_lexical_bundle"]),
        output / "bundle",
        encoder,
        output / "development",
        source,
        job,
        checkpoint=guard,
        product_recipe=recipe,
        accounting_directory=budget,
    )
    verify_bm25_reference(output / "development", Path(spec["frozen_m3_reference"]))
    timings["development_and_exact_m3_seconds"] = time.monotonic() - started
    guard()
    result = {
        "schema_version": 1,
        "status": "passed",
        "rows": coverage["rows"],
        "chunks": coverage["chunks"],
        "artifact_id": bundle_id,
        "timings": timings,
        "M3_reference_exact": True,
        "development_queries": 4129,
        "resource_samples": measurements,
        "test_judgments_accessed": False,
        "products_reencoded": 0,
        "training": False,
        "M5_started": False,
    }
    write_once(output / "CPU_COMPLETE.json", result)
    stage = "persistent_publication"
    for folder in ("bundle", "development"):
        for p in sorted((output / folder).rglob("*")):
            if p.is_file():
                guard()
                target = persistent / output.name / folder / p.relative_to(output / folder)
                resources(spec, publication_bytes=p.stat().st_size)
                target.parent.mkdir(parents=True, exist_ok=True)
                copy_verified(p, target, sha(p))
    for name in ("CPU_COMPLETE.json", "numerical.json", "coverage.json", "progress.jsonl"):
        target = persistent / output.name / name
        resources(spec, publication_bytes=(output / name).stat().st_size)
        copy_verified(output / name, target, sha(output / name))
    receipt = {
        str(p.relative_to(persistent / output.name)): sha(p)
        for p in (persistent / output.name).rglob("*")
        if p.is_file()
    }
    write_once(persistent / output.name / "PUBLISHED.json", {"files": receipt})
    guard()
    event(
        "cpu_full_complete",
        persistent_output=str(persistent / output.name),
        persistent_bytes=size(persistent),
    )


def run(manifest: Path) -> None:
    spec = approved_spec(manifest)
    platform_guard()
    import fcntl

    from atlasrank.retrieval.colab_runner import bounded_command, charged_seconds, size
    from atlasrank.retrieval.colab_transport import once, sha

    persistent = Path(spec["persistent_destination"])
    persistent.mkdir(exist_ok=True)
    owner = {"owner_token": spec["operator_owner_token"], "manifest_sha256": sha(manifest)}
    if (persistent / "OWNER.json").exists():
        if json.loads((persistent / "OWNER.json").read_text()) != owner:
            raise PermissionError("CPU owner/configuration differs")
    elif any(persistent.iterdir()):
        raise PermissionError("Unowned nonempty destination; preserve it")
    else:
        once(persistent / "OWNER.json", owner)
    work = Path(spec["work_directory"])
    work.mkdir(exist_ok=True)
    with (persistent / ".cpu.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        remaining = remaining_seconds(charged_seconds(persistent, 7200))
        attempt = uuid.uuid4().hex
        started = time.monotonic()
        once(
            persistent / (attempt + ".start.json"),
            {"utc_start": time.time(), "reserved_seconds": remaining},
        )
        status = "failed"
        try:
            initial = resources(spec, start=True)
            budget = work / "compute-budget"
            (budget / "attempts").mkdir(parents=True, exist_ok=True)
            if not (budget / "JOB.json").exists():
                once(budget / "JOB.json", {"wall_seconds": 6600})
            deadline = started + remaining
            env = dict(
                os.environ,
                ATLAS_MAC_PARENT_PID=str(os.getpid()),
                ATLAS_MAC_DEADLINE=str(deadline),
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                POLARS_MAX_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                HF_HUB_OFFLINE="1",
                CUDA_VISIBLE_DEVICES="",
            )
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "worker",
                "--manifest",
                str(manifest),
                "--output",
                str(work / ("attempt-" + attempt)),
            ]
            retained_preflight = initial["persistent_new_bytes"] - size(persistent)
            execution_limits = dict(
                spec,
                persistent_limit_bytes=spec["persistent_new_limit_bytes"] - retained_preflight,
                scratch_input_reservation_bytes=initial["scratch_bytes"] - size(work),
            )
            log = work / (attempt + ".log")
            result = bounded_command(
                command, log, deadline, execution_limits, persistent, work, env
            )
            # Include all reviewed CPU inputs/environments and retained preflight dirs.
            resources(spec)
            once(persistent / (attempt + ".command.json"), result)
            status = "passed"
        finally:
            once(
                persistent / (attempt + ".end.json"),
                {
                    "status": status,
                    "elapsed_seconds": time.monotonic() - started,
                    "utc_end": time.time(),
                    "prior_preflight_reservation_seconds": 600,
                },
            )
            if "log" in locals() and log.exists():
                from atlasrank.retrieval.colab_transport import copy_verified

                copy_verified(log, persistent / log.name, sha(log))
        print(
            json.dumps(
                {
                    "status": status,
                    "attempt": attempt,
                    "charged_seconds_including_preflight": 600 + charged_seconds(persistent, 7200),
                }
            ),
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["run", "worker"])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.action == "run":
        run(a.manifest)
    else:
        if a.output is None:
            p.error("Worker output required")
        spec = approved_spec(a.manifest)
        worker(
            a.manifest, Path(spec["work_directory"]), a.output, Path(spec["persistent_destination"])
        )


if __name__ == "__main__":
    main()
