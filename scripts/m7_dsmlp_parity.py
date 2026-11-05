"""Prepared M7 acceptance harness. No submission, training, indexing or evaluation."""

import argparse
import ctypes
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, cast

from m7_dsmlp_preflight import inspect, require_compute


class SupervisorSignal(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"supervisor_signal_{signum}")


@contextmanager
def termination_signals() -> Iterator[None]:
    managed = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous = {sig: signal.getsignal(sig) for sig in managed}

    def interrupt(signum: int, frame: FrameType | None) -> None:
        # Repeated termination must not interrupt the bounded cleanup itself.
        for sig in managed:
            signal.signal(sig, signal.SIG_IGN)
        raise SupervisorSignal(signum)

    for sig in managed:
        signal.signal(sig, interrupt)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def bind_parent_death() -> None:
    expected = int(os.environ.get("ATLAS_M7_PARENT_PID", "-1"))
    if os.getppid() != expected:
        raise ValueError("Worker requires its live resource supervisor")
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    libc.prctl.restype = ctypes.c_int
    # Linux PR_SET_PDEATHSIG: abrupt supervisor death must stop the real worker.
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot bind worker parent-death signal")
    if os.getppid() != expected:
        raise ValueError("Supervisor exited during worker admission")


def validate_execution(config: dict[str, Any]) -> None:
    required = {
        "schema_version": 1,
        "native_math_threads": 1,
        "approved": True,
        "scope": "M7_real_model_http_parity",
        "cpus": 4,
        "ram_gb": 32,
        "gpus": 0,
        "wall_seconds": 300,
        "rss_bytes": 12 * 1024**3,
        "minimum_available_bytes": 8 * 1024**3,
    }
    if any(type(config.get(k)) is not type(v) or config.get(k) != v for k, v in required.items()):
        raise ValueError("Approval and the exact bounded M7 CPU execution contract are required")


def write(path: Path, value: object) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def capture(value: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return {
        "names": value.names,
        "feature_types": value.feature_types,
        "keys": value.keys,
        "values": value.values.tolist(),
        "dtype": str(value.values.dtype),
        "missing": value.missing.tolist(),
        "text_empty": value.text_empty.tolist(),
        "provenance": asdict(value.provenance),
        "schema_id": value.schema_id,
        "query_fingerprint": value.query_fingerprint,
    }


def worker(args: argparse.Namespace) -> int:
    bind_parent_death()
    import uvicorn

    from atlasrank.artifacts.dense import file_hash
    from atlasrank.data.common import peak_rss_bytes
    from atlasrank.evaluation.ranking import offline_predict
    from atlasrank.ranking.learned import FittedPredictor
    from atlasrank.ranking.prediction import assert_exact_parity
    from atlasrank.retrieval.fusion import reciprocal_rank_fusion
    from atlasrank.serving.app import create_app
    from atlasrank.serving.runtime import DenseSearch, load_runtime

    started = time.monotonic()
    reference = (
        json.loads(args.original_query_recipe.read_text())
        if args.allow_platform_difference
        else None
    )
    runtime = load_runtime(args.descriptor, args.artifacts, reference_query_recipe=reference)
    context = runtime.context
    if context is None or runtime.ranking is None:
        raise ValueError("Full real-model runtime required")
    ranker = runtime.ranking.ranker
    if not isinstance(ranker.model, FittedPredictor) or not ranker.model.trained:
        raise ValueError("Deterministic/zero-update substitutes cannot satisfy M7")
    model = ranker.model
    model.record = True
    spec = json.loads((args.descriptor / "spec.json").read_text())
    model_path = args.artifacts / spec["components"]["ranker"] / "model.ubj"
    model_before = file_hash(model_path)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(serving_runtime=runtime),
            host="127.0.0.1",
            port=port,
            workers=1,
            access_log=False,
            log_config=None,
            timeout_graceful_shutdown=5,
            limit_concurrency=8,
            backlog=16,
            timeout_keep_alive=5,
        )
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def request(path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode() if body else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return dict(json.load(response))

    observations = []
    first_capture = None
    any_zero = False
    ready_seconds = None
    shutdown_ok = False
    try:
        ready_deadline = min(started + 270, time.monotonic() + 30)
        while time.monotonic() < ready_deadline:
            try:
                readiness = request("/health/ready")
                if (
                    readiness.get("ready") is True
                    and readiness.get("bundle_id") == runtime.bundle_id
                ):
                    ready_seconds = time.monotonic() - started
                    break
            except OSError:
                pass
            if not thread.is_alive():
                raise RuntimeError("HTTP server stopped before readiness")
            time.sleep(0.1)
        if ready_seconds is None:
            raise RuntimeError("Readiness deadline exceeded")
        # Newly declared label-free smoke queries, never presented as original M6 cases.
        queries = [
            "wireless headphones",
            " ＷＩＲＥＬＥＳＳ\t headphones ",
            "zzatlasmsevenunmatchedtokenzz",
        ]
        blank = next(
            (
                key
                for key, title in zip(context.lexical.keys, context.lexical.titles, strict=True)
                if not title.strip()
            ),
            None,
        )
        for number, query in enumerate(queries):
            lexical = [(h.product_locale, h.product_id) for h in context.lexical.search(query, 100)]
            dense = [
                key
                for key, _ in cast(DenseSearch, context.dense).search(
                    context.encoder.encode([query]), 100
                )
            ]
            pool = [key for key, _ in reciprocal_rank_fusion([lexical, dense], 100)]
            if not pool:
                raise ValueError("Expected actual dense candidate pool")
            model.observations.clear()
            expected = offline_predict(context, ranker, query, pool)
            response = request("/search", {"query": query, "pipeline": "m6_xgb_m4", "top_k": 100})
            if len(model.observations) != 2:
                raise ValueError("Missing actual native captures")
            a, b = model.observations
            assert_exact_parity(a, b)
            if number == 0:
                first_capture = a
            elif number == 1:
                if first_capture is None:
                    raise ValueError("Missing normalization reference capture")
                assert_exact_parity(first_capture, a)
            actual = [
                ((r["product_locale"], r["product_id"]), r["score"]) for r in response["results"]
            ]
            if actual != expected or {key for key, _ in actual} != set(pool):
                raise ValueError("Full HTTP predictions/order/candidate membership differs")
            if (
                response.get("execution") != "live"
                or response.get("experimental") is not True
                or response.get("versions", {}).get("ranker") != ranker.artifact_id
            ):
                raise ValueError("Incorrect HTTP model/execution identity")
            outside = next(key for key in context.lexical.keys if key not in set(lexical + dense))
            fixed = sorted(set(pool[:6] + [outside] + ([blank] if blank else [])))
            model.observations.clear()
            fixed_expected = offline_predict(context, ranker, query, fixed)
            fixed_response = request(
                "/rerank",
                {"query": query, "top_k": 100, "product_ids": [k[1] for k in reversed(fixed)]},
            )
            if len(model.observations) != 2:
                raise ValueError("Missing fixed-candidate native captures")
            c, d = model.observations
            assert_exact_parity(c, d)
            fixed_actual = [
                ((r["product_locale"], r["product_id"]), r["score"])
                for r in fixed_response["results"]
            ]
            if (
                fixed_response.get("learned") is not True
                or fixed_response.get("mode") != "fixed_candidate_learned"
                or fixed_response.get("model_id") != ranker.artifact_id
            ):
                raise ValueError("Incorrect fixed-candidate learned-model identity")
            any_zero = any_zero or bool((c.values[:, 0] == 0).any())
            if fixed_expected != fixed_actual:
                raise ValueError("Permuted HTTP fixed-candidate output differs")
            record = {
                "case": number,
                "query": query,
                "new_M7_case_not_original_M6": True,
                "search_offline": capture(a),
                "search_http": capture(b),
                "rerank_offline": capture(c),
                "rerank_http": capture(d),
                "search_response": response,
                "rerank_response": fixed_response,
                "candidate_membership_exact": True,
                "predictions_exact": True,
                "post_dmatrix_exact": True,
                "genuine_zero_bm25": bool((c.values[:, 0] == 0).any()),
                "missing_text_available": blank is not None,
                "absent_top100_product": outside,
            }
            write(args.output / f"case-{number:02d}.json", record)
            observations.append({"case": number, "status": "passed", "pool_size": len(pool)})
        if not any_zero:
            raise ValueError("Actual genuine zero-BM25 coverage not established")
        if file_hash(model_path) != model_before:
            raise ValueError("Original model bytes changed")
    finally:
        model.record = False
        model.observations.clear()
        server.should_exit = True
        thread.join(timeout=8)
        shutdown_ok = not thread.is_alive()
        sock.close()
    if not shutdown_ok:
        raise RuntimeError("HTTP server did not shut down within reserve")
    write(
        args.output / "worker-result.json",
        {
            "status": "passed_real_model_http_parity",
            "cases": observations,
            "startup_to_readiness_seconds": ready_seconds,
            "worker_wall_seconds": time.monotonic() - started,
            "reported_peak_process_rss_bytes": peak_rss_bytes(),
            "shutdown_ok": shutdown_ok,
            "bundle_id": runtime.bundle_id,
            "model_id": ranker.artifact_id,
            "model_sha256_before_after": model_before,
            "historical_query_recipe_id": spec["query_recipe_id"],
            "actual_query_recipe_id": context.provenance.query_recipe_id,
            "actual_query_recipe": context.encoder.recipe,
            "platform_difference_allowed": args.allow_platform_difference,
            "original_artifacts_modified": False,
            "development_or_test_evaluation": False,
            "new_fit_count": 0,
        },
    )
    return 0


def stop_process(process: subprocess.Popen[Any]) -> int:
    # The root can exit before its descendants; always address our owned group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        result = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        result = process.wait(timeout=2)
    # Don't leave descendants behind after the group root has already stopped.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return result


def supervise(args: argparse.Namespace, config: dict[str, Any]) -> int:
    import psutil

    validate_execution(config)
    if args.output.exists():
        raise ValueError("Preserve previous output directories")
    expected = json.loads((args.descriptor / "spec.json").read_text())["query_recipe_id"]
    preflight = inspect(args.original_query_recipe, expected, args.allow_platform_difference)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "preflight.json", preflight)
    if preflight["status"] != "passed_metadata_preflight_only":
        return 2
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--execution-manifest",
        str(args.execution_manifest),
        "--descriptor",
        str(args.descriptor),
        "--artifacts",
        str(args.artifacts),
        "--original-query-recipe",
        str(args.original_query_recipe),
        "--output",
        str(args.output),
    ]
    if args.allow_platform_difference:
        command.append("--allow-platform-difference")
    env = dict(
        os.environ,
        ATLAS_M7_PARENT_PID=str(os.getpid()),
        PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    env.pop("PYTHONPATH", None)
    started = time.monotonic()
    peak = 0
    reason = None
    with (args.output / "worker.log").open("x") as log, termination_signals():
        process = None
        exit_code = -1
        try:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                try:
                    root = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > 12 * 1024**3:
                    reason = "peak_rss_limit"
                if time.monotonic() - started >= 290:
                    reason = "wall_limit_with_10_second_shutdown_reserve"
                if reason:
                    break
                time.sleep(0.1)
        except SupervisorSignal as error:
            reason = str(error)
        except (OSError, psutil.Error) as error:
            reason = "supervisor_monitor_failed:" + type(error).__name__
        finally:
            if process is not None:
                exit_code = stop_process(process)
    elapsed = time.monotonic() - started
    worker_result = args.output / "worker-result.json"
    ok = (
        reason is None
        and exit_code == 0
        and elapsed <= 300
        and peak <= 12 * 1024**3
        and worker_result.is_file()
        and json.loads(worker_result.read_text()).get("status") == "passed_real_model_http_parity"
        and json.loads(worker_result.read_text()).get(
            "reported_peak_process_rss_bytes", 13 * 1024**3
        )
        <= 12 * 1024**3
    )
    report = {
        "status": "passed" if ok else "failed_preserved_no_retry",
        "reason": reason,
        "wall_seconds": elapsed,
        "wall_limit_seconds": 300,
        "sampled_peak_tree_rss_bytes": peak,
        "rss_limit_bytes": 12 * 1024**3,
        "sample_interval_seconds": 0.1,
        "worker_exit_code": exit_code,
        "shutdown_reserve_seconds": 10,
        "platform_difference_allowed": args.allow_platform_difference,
        "historical_query_recipe_id": expected,
        "actual_query_recipe_id": preflight["actual_query_recipe_id"],
        "execution_manifest_sha256": hashlib.sha256(
            args.execution_manifest.read_bytes()
        ).hexdigest(),
    }
    write(args.output / "supervisor.json", report)
    # Hash evidence after termination, outside the measured smoke; never overwrite inputs.
    write(
        args.output / "checksums.json",
        {
            str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.iterdir()
            if p.is_file()
        },
    )
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--original-query-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--allow-platform-difference", action="store_true")
    args = parser.parse_args()
    require_compute(
        socket.gethostname(), sys.platform, bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    )
    config = json.loads(args.execution_manifest.read_text())
    validate_execution(config)
    if (
        args.allow_platform_difference
        and config.get("compatibility_policy") != "record_linux_platform_difference"
    ):
        raise ValueError("Execution manifest must declare platform portability")
    return worker(args) if args.worker else supervise(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
