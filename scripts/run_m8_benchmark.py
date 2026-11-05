"""Supervise a bounded M8 CPU reference server and separate HTTP load processes."""

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from m7_dsmlp_parity import bind_parent_death, stop_process, termination_signals
from m7_dsmlp_preflight import inspect, require_compute, resources


def write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_execution(config: dict[str, Any]) -> None:
    fixed = {
        "schema_version": 1,
        "approved": True,
        "scope": "M8_CPU_reference_benchmark",
        "cpus": 4,
        "ram_gb": 32,
        "gpus": 0,
        "wall_seconds": 14400,
        "rss_bytes": 12 * 1024**3,
        "minimum_available_bytes": 8 * 1024**3,
        "new_storage_bytes": 1024**3,
        "free_headroom_bytes": 2 * 1024**3,
        "native_math_threads": 1,
        "warmup_requests": 100,
        "measured_requests": 1000,
        "repeats": 3,
        "request_timeout_seconds": 30,
        "compatibility_policy": "record_linux_platform_difference",
    }
    if any(type(config.get(k)) is not type(v) or config.get(k) != v for k, v in fixed.items()):
        raise ValueError("Explicit approval and the fixed M8 execution contract are required")
    if config.get("concurrency") != [1, 4, 8] or config.get("pipelines") != [
        "bm25",
        "rrf_m4",
        "m6_xgb_m4",
    ]:
        raise ValueError("Reference workload or pipelines differ")
    if type(config.get("execution_deadline_utc")) not in (int, float) or not math.isfinite(
        config["execution_deadline_utc"]
    ):
        raise ValueError("External cumulative execution deadline is required")


def verify_source_snapshot(root: Path, registered_manifest_sha256: str) -> None:
    manifest = root / "snapshot-manifest.json"
    if digest_file(manifest) != registered_manifest_sha256:
        raise ValueError("Registered source manifest hash mismatch")
    for record in read(root / "snapshot-manifest.json")["files"]:
        relative = Path(record["path"])
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
            raise ValueError("Unsafe source snapshot path")
        if digest_file(path) != record["sha256"]:
            raise ValueError("Source snapshot hash mismatch")


def storage_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def request(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return dict(json.load(response))


def serve(args: argparse.Namespace) -> int:
    started = time.monotonic()
    bind_parent_death()
    import uvicorn

    from atlasrank.benchmarking.profiling import ProfiledRuntime
    from atlasrank.data.common import peak_rss_bytes
    from atlasrank.evaluation.ranking import offline_predict
    from atlasrank.ranking.learned import FittedPredictor
    from atlasrank.ranking.prediction import assert_exact_parity
    from atlasrank.serving.app import create_app
    from atlasrank.serving.runtime import load_runtime

    runtime = ProfiledRuntime(
        load_runtime(
            args.descriptor, args.artifacts, reference_query_recipe=read(args.reference_recipe)
        )
    )
    if runtime.context is None or runtime.ranking is None:
        raise ValueError("A full real-model reference is required")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base = "http://127.0.0.1:" + str(sock.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(serving_runtime=runtime),
            host="127.0.0.1",
            workers=1,
            access_log=False,
            log_config=None,
            limit_concurrency=8,
            backlog=16,
            timeout_keep_alive=5,
            timeout_graceful_shutdown=5,
        )
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    shutdown = False
    spec = read(args.descriptor / "spec.json")
    model = args.artifacts / spec["components"]["ranker"] / "model.ubj"
    model_before = digest_file(model)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                ready = request(base + "/health/ready")
                if ready.get("ready") is True:
                    break
            except OSError:
                pass
            if not thread.is_alive():
                raise RuntimeError("Reference HTTP process stopped")
            time.sleep(0.1)
        else:
            raise TimeoutError("HTTP readiness failed")
        readiness_seconds = time.monotonic() - started
        # Check the instrumented shared path against actual HTTP before load begins.
        predictor = runtime.ranking.ranker.model
        if not isinstance(predictor, FittedPredictor):
            raise ValueError("The real fitted model is required")
        predictor.record = True
        predictor.observations.clear()
        query = "wireless headphones"
        response = request(
            base + "/search", {"query": query, "pipeline": "m6_xgb_m4", "top_k": 100}
        )
        keys = [(row["product_locale"], row["product_id"]) for row in response["results"]]
        expected = offline_predict(runtime.context, runtime.ranking.ranker, query, keys)
        assert_exact_parity(predictor.observations[0], predictor.observations[1])
        actual = [(key, row["score"]) for key, row in zip(keys, response["results"], strict=True)]
        if actual != expected:
            raise ValueError("Instrumented HTTP prediction parity differs")
        predictor.record = False
        predictor.observations.clear()
        write(
            args.output / "server-ready.json",
            {
                "url": base,
                "readiness_response": ready,
                "startup_to_readiness_seconds": readiness_seconds,
                "instrumented_real_http_parity": "passed_exact_features_and_predictions",
                "model_sha256_before": model_before,
                "actual_query_recipe": runtime.context.encoder.recipe,
                "versions": response["versions"],
            },
        )
        execution_deadline = read(args.execution_manifest)["execution_deadline_utc"]
        while not (args.output / "stop-server").exists():
            if time.time() >= execution_deadline:
                raise TimeoutError("Execution deadline reached")
            if not thread.is_alive():
                raise RuntimeError("Reference HTTP thread exited")
            time.sleep(0.2)
        if digest_file(model) != model_before:
            raise ValueError("Immutable model hash changed")
    finally:
        server.should_exit = True
        thread.join(timeout=8)
        shutdown = not thread.is_alive()
        sock.close()
        write(
            args.output / "server-shutdown.json",
            {
                "shutdown_ok": shutdown,
                "model_sha256_after": digest_file(model),
                "reported_peak_process_rss_bytes": peak_rss_bytes(),
                "wall_seconds": time.monotonic() - started,
            },
        )
    return 0 if shutdown else 2


def supervise(args: argparse.Namespace) -> int:
    import psutil

    config = read(args.execution_manifest)
    validate_execution(config)
    if args.output.exists():
        raise ValueError("Preserve existing evidence; choose a fresh run directory")
    deadline = min(float(config["execution_deadline_utc"]), time.time() + 14400)
    if deadline - time.time() < 60:
        raise ValueError("Insufficient approved time remains")
    args.output.mkdir(parents=True, exist_ok=False)
    expected = read(args.descriptor / "spec.json")["query_recipe_id"]
    preflight = inspect(args.reference_recipe, expected, allow_platform_difference=True)
    write(args.output / "preflight.json", preflight)
    if preflight["status"] != "passed_metadata_preflight_only":
        return 2
    workload = read(args.workload)
    if len(workload["queries"]) != 1000 or workload.get("partition") != "development":
        raise ValueError("Exactly 1000 fixed development queries required")
    if digest_file(args.workload) != config["workload_sha256"]:
        raise ValueError("Workload bytes differ from registered input")
    verify_source_snapshot(args.source_root, config["source_manifest_sha256"])
    descriptor = read(args.descriptor / "spec.json")
    dense_dir = args.artifacts / descriptor["components"]["dense"]
    write(
        args.output / "environment.json",
        {
            "hostname": socket.gethostname(),
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unavailable",
            ),
            "cpu_affinity": psutil.Process().cpu_affinity(),
            "load_average": os.getloadavg(),
            "power_conditions": "shared_cluster_uncontrolled",
            "index_bytes": (dense_dir / "index.faiss").stat().st_size,
            "execution_manifest_sha256": digest_file(args.execution_manifest),
            "workload_sha256": digest_file(args.workload),
            "source_manifest_sha256": digest_file(args.source_root / "snapshot-manifest.json"),
            "disk_headroom_basis": (
                "minimum of filesystem free and conservative 4GiB quota baseline minus new bytes"
            ),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "POLARS_MAX_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
        },
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(args.source_root / "src"),
        PYTHONNOUSERSITE="1",
        ATLAS_M7_PARENT_PID=str(os.getpid()),
    )
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--server"]
    started = time.monotonic()
    peak = 0
    server_process = None
    client = None
    trials: list[dict[str, Any]] = []
    reason = None

    def monitor(*, allow_server_exit: bool = False) -> None:
        nonlocal peak
        supervisor_process = psutil.Process()
        processes = supervisor_process.children(recursive=True)
        rss = supervisor_process.memory_info().rss
        for process in processes:
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        peak = max(peak, rss)
        added = storage_bytes(args.budget_root)
        headroom = min(shutil.disk_usage(args.budget_root).free, 4 * 1024**3 - added)
        if rss > config["rss_bytes"]:
            raise RuntimeError("Aggregate process-tree RSS limit")
        if resources()["effective_available_bytes"] < config["minimum_available_bytes"]:
            raise RuntimeError("Effective available RAM fell below admission reserve")
        if added > config["new_storage_bytes"] or headroom < config["free_headroom_bytes"]:
            raise RuntimeError("Storage cap or free-headroom limit")
        if time.time() >= deadline - 40:
            raise TimeoutError("Cumulative M8 execution deadline")
        if (
            not allow_server_exit
            and server_process is not None
            and server_process.poll() is not None
        ):
            raise RuntimeError("Reference server exited before completion")

    with (args.output / "server.log").open("x") as log, termination_signals():
        try:
            server_process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while not (args.output / "server-ready.json").exists():
                monitor()
                time.sleep(0.1)
            ready = read(args.output / "server-ready.json")
            for repeat in range(3):
                for concurrency in (1, 4, 8):
                    pipelines = config["pipelines"][repeat:] + config["pipelines"][:repeat]
                    for pipeline in pipelines:
                        trial = args.output / f"{pipeline}-c{concurrency}-r{repeat}"
                        cmd = [
                            sys.executable,
                            "-m",
                            "atlasrank.benchmarking.load",
                            "--url",
                            ready["url"] + "/search",
                            "--workload",
                            str(args.workload),
                            "--output",
                            str(trial),
                            "--pipeline",
                            pipeline,
                            "--concurrency",
                            str(concurrency),
                            "--repeat",
                            str(repeat),
                            "--requests",
                            "1000",
                            "--warmup",
                            "100",
                            "--timeout",
                            "30",
                            "--deadline-utc",
                            str(deadline - 40),
                            "--expected-bundle",
                            ready["readiness_response"]["bundle_id"],
                        ]
                        with (args.output / "client.log").open("a") as client_log:
                            client = subprocess.Popen(
                                cmd,
                                env=env,
                                stdout=client_log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            while client.poll() is None:
                                monitor()
                                time.sleep(0.1)
                        if client.returncode != 0:
                            raise RuntimeError("Load generator stopped; retain partial trial")
                        summary = read(trial / "summary.json")
                        trials.append(summary)
                        print(
                            json.dumps(
                                {
                                    "event": "trial_completed",
                                    "pipeline": pipeline,
                                    "concurrency": concurrency,
                                    "repeat": repeat,
                                    "succeeded": summary["counts"]["succeeded"],
                                    "errors": summary["counts"]["errors"],
                                }
                            ),
                            flush=True,
                        )
            (args.output / "stop-server").touch(exist_ok=False)
            graceful_deadline = time.monotonic() + 12
            while server_process.poll() is None:
                monitor(allow_server_exit=True)
                if time.monotonic() >= graceful_deadline:
                    raise TimeoutError("Graceful reference server shutdown exceeded reserve")
                time.sleep(0.1)
        except BaseException as error:
            reason = type(error).__name__ + ":" + str(error)
        finally:
            if client is not None:
                stop_process(client)
            if server_process is not None:
                if server_process.poll() is None:
                    (args.output / "stop-server").touch(exist_ok=True)
                    try:
                        server_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        stop_process(server_process)
    shutdown_path = args.output / "server-shutdown.json"
    shutdown = read(shutdown_path) if shutdown_path.exists() else {}
    passed = (
        reason is None
        and len(trials) == 27
        and shutdown.get("shutdown_ok") is True
        and shutdown.get("reported_peak_process_rss_bytes", 13 * 1024**3) <= config["rss_bytes"]
        and server_process is not None
        and server_process.returncode == 0
    )
    write(
        args.output / "result.json",
        {
            "status": "completed_reference_benchmark" if passed else "failed_preserved",
            "reason": reason,
            "trials": trials,
            "sampled_peak_tree_rss_bytes": peak,
            "sample_interval_seconds": 0.1,
            "shutdown_reserve_seconds": 40,
            "wall_seconds": time.monotonic() - started,
            "new_storage_bytes": storage_bytes(args.budget_root),
            "shutdown": shutdown,
            "server_exit_code": server_process.returncode if server_process else None,
        },
    )
    write(
        args.output / "checksums.json",
        {
            str(path.relative_to(args.output)): digest_file(path)
            for path in args.output.rglob("*")
            if path.is_file()
        },
    )
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "execution-manifest",
        "descriptor",
        "artifacts",
        "reference-recipe",
        "workload",
        "source-root",
        "budget-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()
    require_compute(
        socket.gethostname(), sys.platform, bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    )
    validate_execution(read(args.execution_manifest))
    return serve(args) if args.server else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
