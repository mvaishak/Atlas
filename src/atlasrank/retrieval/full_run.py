"""Gated CPU-only full-run supervisor. A proposed manifest never authorizes execution."""

import argparse
import fcntl
import gc
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from atlasrank.artifacts.dense import InvalidDense, file_hash, write_once
from atlasrank.data.common import check_disk, tree_bytes
from atlasrank.retrieval.chunks import accounted_seconds, canonical_catalog, encode_chunks
from atlasrank.retrieval.comparison import evaluate_development, verify_bm25_reference
from atlasrank.retrieval.dense_bundle import build_bundle, load_bundle
from atlasrank.retrieval.encoder import PinnedEncoder
from atlasrank.retrieval.encoder import configuration as dense_configuration
from atlasrank.retrieval.jobs import configuration as lexical_configuration


def implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = [*sorted((root / "src/atlasrank").rglob("*.py")), root / "uv.lock"]
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def approved_manifest(path: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text())
    if spec.get("execution_approved") is not True:
        raise PermissionError("Full-catalog execution requires explicit lead resource approval")
    frozen = {
        "schema_version": 1,
        "pipeline": "m4-cpu-reference-v1",
        "data_root": "data/esci",
        "wall_seconds": 43200,
        "rss_limit_bytes": 8 * 1024**3,
        "additional_artifact_limit_bytes": 8 * 1024**3,
        "available_ram_start_bytes": 6 * 1024**3,
        "available_ram_stop_bytes": 1024**3,
    }
    if any(spec.get(k) != v for k, v in frozen.items()):
        raise ValueError("Unreviewed full-run budget or pipeline")
    if spec["implementation_sha256"] != implementation_hashes():
        raise ValueError("Full-run implementation/lock differs from reviewed manifest")
    for name in ("dense_config", "lexical_config", "evaluation_contract"):
        if file_hash(Path(spec[name])) != spec[name + "_sha256"]:
            raise ValueError("Full-run source configuration differs")
    if file_hash(Path(spec["bm25_reference_per_query"])) != spec["bm25_reference_per_query_sha256"]:
        raise ValueError("Frozen M3 reference output differs")
    root, run = Path(spec["data_root"]), Path(spec["run_directory"])
    if not run.resolve().is_relative_to(root.resolve()) or run.resolve() == root.resolve():
        raise ValueError("Run must have its own directory inside the budgeted data root")
    return spec


def supervise(path: Path) -> Path:
    entered = time.monotonic()
    utc_entered = datetime.now(UTC).timestamp()
    spec = approved_manifest(path)  # Fail before any data/model/process work when unapproved.
    root, run = Path(spec["data_root"]), Path(spec["run_directory"])
    if psutil.virtual_memory().available < spec["available_ram_start_bytes"]:
        raise RuntimeError("Full run requires at least 6 GiB available system RAM")
    check_disk(
        root,
        max(0, spec["additional_artifact_limit_bytes"] - (tree_bytes(run) if run.exists() else 0)),
    )
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".supervisor.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A worker can outlive an abruptly killed supervisor. Don't start a
        # second worker or recover its accounting while the old worker is active.
        with (run / ".worker.lock").open("a+b") as worker_lock:
            try:
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Previous full-run worker remains active") from error
        if (run / "RUN.json").exists():
            if json.loads((run / "RUN.json").read_text()) != spec:
                raise ValueError("Resume run manifest differs")
        else:
            write_once(run / "RUN.json", spec)
        ledger = run / "supervisor"
        (ledger / "attempts").mkdir(parents=True, exist_ok=True)
        charged = accounted_seconds(ledger)
        remaining = spec["wall_seconds"] - charged
        if remaining <= 0:
            raise RuntimeError("Cumulative full-run budget exhausted")
        attempt = uuid.uuid4().hex
        write_once(
            ledger / "attempts" / f"{attempt}.start.json",
            {"utc_start": utc_entered, "reserved_seconds": remaining},
        )
        environment = dict(
            os.environ,
            POLARS_MAX_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1",
            ATLAS_M4_SUPERVISOR_PID=str(os.getpid()),
            ATLAS_M4_DEADLINE_MONOTONIC=str(entered + spec["wall_seconds"] - charged),
        )
        status, detail, peak = "failed", "", 0
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "atlasrank.retrieval.full_run",
                "--worker",
                str(run / "RUN.json"),
            ],
            env=environment,
        )
        try:
            process = psutil.Process(child.pid)
            last_disk_check = 0.0
            while child.poll() is None:
                if charged + time.monotonic() - entered >= spec["wall_seconds"]:
                    raise RuntimeError("12-hour cumulative full-run command cap exceeded")
                try:
                    rss = process.memory_info().rss
                except psutil.NoSuchProcess:
                    break
                peak = max(peak, rss)
                if (
                    rss > spec["rss_limit_bytes"]
                    or psutil.virtual_memory().available < spec["available_ram_stop_bytes"]
                ):
                    raise RuntimeError("Full-run memory guard exceeded")
                if time.monotonic() - last_disk_check >= 1:
                    check_disk(root)
                    if tree_bytes(run) > spec["additional_artifact_limit_bytes"]:
                        raise RuntimeError("8 GiB incremental run artifact cap exceeded")
                    last_disk_check = time.monotonic()
                time.sleep(0.1)
            if child.wait():
                raise RuntimeError("Full-run worker failed; artifacts preserved")
            check_disk(root)
            if tree_bytes(run) > spec["additional_artifact_limit_bytes"]:
                raise RuntimeError("Full-run artifact cap exceeded at completion")
            status = "passed"
        except BaseException as error:
            detail = str(error)
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            write_once(
                ledger / "attempts" / f"{attempt}.end.json",
                {
                    "status": status,
                    "detail": detail,
                    "elapsed_seconds": time.monotonic() - entered,
                    "sampled_peak_worker_rss_bytes": peak,
                },
            )
    return run


def worker(path: Path) -> None:
    spec = approved_manifest(path)
    with (Path(spec["run_directory"]) / ".worker.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _worker(spec)


def worker_guard(spec: dict[str, Any]) -> Callable[[], None]:
    parent = int(os.environ.get("ATLAS_M4_SUPERVISOR_PID", "0"))
    deadline = float(os.environ.get("ATLAS_M4_DEADLINE_MONOTONIC", "0"))
    last_resources = 0.0

    def guard() -> None:
        nonlocal last_resources
        if parent <= 0 or os.getppid() != parent or not psutil.pid_exists(parent):
            raise RuntimeError("Full worker lost its expected supervisor")
        if time.monotonic() >= deadline:
            raise RuntimeError("Full worker cumulative deadline reached")
        if time.monotonic() - last_resources >= 1:
            if (
                psutil.Process().memory_info().rss > spec["rss_limit_bytes"]
                or psutil.virtual_memory().available < spec["available_ram_stop_bytes"]
            ):
                raise RuntimeError("Full worker memory guard exceeded")
            check_disk(Path(spec["data_root"]))
            if tree_bytes(Path(spec["run_directory"])) > spec["additional_artifact_limit_bytes"]:
                raise RuntimeError("Full worker artifact cap exceeded")
            last_resources = time.monotonic()

    return guard


def _worker(spec: dict[str, Any]) -> None:
    guard = worker_guard(spec)
    guard()
    root, run = Path(spec["data_root"]), Path(spec["run_directory"])
    lexical = lexical_configuration(Path(spec["lexical_config"]), root)
    dense = dense_configuration(Path(spec["dense_config"]))
    if any(
        dense[k] != lexical[k] for k in ("prepared_path", "prepared_audit_sha256", "catalog_sha256")
    ):
        raise InvalidDense("Dense/lexical source preparation differs")
    canonical = run / "canonical.parquet"
    if canonical.exists():
        record = json.loads((run / "CANONICAL.json").read_text())
        if record != {
            "original_catalog_sha256": lexical["catalog_sha256"],
            "canonical_sha256": file_hash(canonical),
        }:
            raise InvalidDense("Canonical catalog differs; preserve previous files")
    else:
        canonical_catalog(Path(lexical["prepared_path"]) / "catalog.parquet", canonical)
        write_once(
            run / "CANONICAL.json",
            {
                "original_catalog_sha256": lexical["catalog_sha256"],
                "canonical_sha256": file_hash(canonical),
            },
        )
    encoder = PinnedEncoder(dense)
    guard()
    recipe = encoder.recipe
    origin = {k: lexical[k] for k in ("prepared_audit_sha256", "catalog_sha256")}
    origin["scope"] = "approved_us_catalog"
    job = run / "embeddings"
    encode_chunks(
        canonical,
        job,
        encoder,
        origin,
        chunk_rows=1024,
        wall_seconds=spec["wall_seconds"],
        budget_root=root,
        checkpoint=guard,
    )
    del encoder
    gc.collect()
    bundles = run / "bundles"
    bundles.mkdir(exist_ok=True)
    if (run / "BUNDLE.json").exists():
        bundle = run / json.loads((run / "BUNDLE.json").read_text())["relative_path"]
        load_bundle(bundle, recipe, expected_origin=origin)
    else:
        complete_bundles = sorted(p for p in bundles.iterdir() if (p / "COMPLETE.json").exists())
        bundle = complete_bundles[0] if complete_bundles else bundles / uuid.uuid4().hex
        if not complete_bundles:
            build_bundle(job, bundle, budget_root=root, checkpoint=guard)
        load_bundle(bundle, recipe, expected_origin=origin)
        write_once(run / "BUNDLE.json", {"relative_path": str(bundle.relative_to(run))})
    encoder = PinnedEncoder(dense)
    guard()
    pins = {
        k: lexical[k] for k in ("catalog_sha256", "prepared_audit_sha256", "development_sha256")
    }
    pins["scope"] = "approved_us_catalog"
    reports = run / "reports"
    reports.mkdir(exist_ok=True)
    if (run / "REPORT.json").exists():
        report_path = run / json.loads((run / "REPORT.json").read_text())["relative_path"]
        if (
            file_hash(report_path / "report.json")
            != json.loads((report_path / "COMPLETE.json").read_text())["report_sha256"]
        ):
            raise InvalidDense("Completed report differs")
        verify_bm25_reference(report_path, Path(spec["bm25_reference_per_query"]))
    else:
        # A partial evaluation is retained; resume recomputes the whole fixed suite.
        destination = reports / uuid.uuid4().hex
        evaluate_development(
            Path(lexical["prepared_path"]),
            Path(lexical["index_path"]),
            bundle,
            encoder,
            destination,
            pins,
            job,
            checkpoint=guard,
        )
        verify_bm25_reference(destination, Path(spec["bm25_reference_per_query"]))
        write_once(run / "REPORT.json", {"relative_path": str(destination.relative_to(run))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    worker(args.worker)
