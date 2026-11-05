"""Resource-supervised, single-attempt real M4 ranker experiment. Linux CPU only."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil


def tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rss-limit", type=int, default=12 * 1024**3)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("No native ranker execution on the Mac")
    if not 1 <= args.wall_seconds <= 7200 or not 0 < args.rss_limit <= 12 * 1024**3:
        parser.error("Resource limits cannot exceed the approved ceilings")
    if args.worker:
        from atlasrank.ranking.experiment import run

        return run(args.inputs, args.source, args.output, args.wall_seconds, args.rss_limit)
    if args.output.exists():
        parser.error("Preserve existing experiment directories; no full-fit retries")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    available = psutil.virtual_memory().available
    rss_limit = min(args.rss_limit, available - 1024**3)
    report: dict[str, Any] = {
        "start_available_ram_bytes": available,
        "worker_rss_limit_bytes": rss_limit,
        "wall_seconds_limit": args.wall_seconds,
        "native_process_started": False,
    }
    reason = None
    if args.wall_seconds < 300:
        reason = "insufficient_wall_budget_for_native_gate_and_real_workload"
    if available < 8 * 1024**3:
        reason = "insufficient_measured_start_ram_below_8_gib_for_real_bundle_with_headroom"
    if tree_bytes(args.scratch_root) > 8 * 1024**3:
        reason = "scratch_limit_at_start"
    if psutil.disk_usage(args.scratch_root).free < 512 * 1024**2:
        reason = "insufficient_free_disk"
    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        reason = "unsafe_native_override_present"
    if reason:
        report["status"] = reason
        (args.output / "supervisor.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return 2
    env = dict(
        os.environ,
        PYTHONPATH=str(args.source.resolve() / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    # Re-run the native fixture gate in its own fresh process, without importing Torch.
    native_command = [
        sys.executable,
        str(args.source / "scripts/check_xgboost_boundary.py"),
        "--output",
        str(args.output / "native-fixture"),
    ]
    with (args.output / "native-fixture.log").open("x") as log:
        native = subprocess.Popen(
            native_command,
            env=env,
            cwd=args.source,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            native.wait(timeout=210)
        except subprocess.TimeoutExpired:
            processes = [psutil.Process(native.pid)]
            processes.extend(processes[0].children(recursive=True))
            for child in reversed(processes):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            native.wait(timeout=10)
            report["status"] = "native_fixture_timeout_process_tree_stopped_no_retry"
            with (args.output / "supervisor.json").open("x") as saved:
                json.dump(report, saved, indent=2)
            return 2
    if native.returncode:
        report["status"] = "native_fixture_gate_failed_no_retry"
        report["exit_code"] = native.returncode
        (args.output / "supervisor.json").write_text(json.dumps(report, indent=2) + "\n")
        return 2
    remaining = args.wall_seconds - int(time.monotonic() - started)
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--worker",
        "--inputs",
        str(args.inputs),
        "--source",
        str(args.source),
        "--output",
        str(args.output),
        "--scratch-root",
        str(args.scratch_root),
        "--wall-seconds",
        str(remaining),
        "--rss-limit",
        str(rss_limit),
    ]
    report["worker_command"] = command
    peak_rss = 0
    max_scratch = 0
    max_persistent = 0
    lowest_available = available
    next_disk_check = 0.0
    report["native_process_started"] = True
    with (args.output / "worker.log").open("x") as log:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=args.source,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            try:
                child = psutil.Process(process.pid)
                rss = child.memory_info().rss + sum(
                    p.memory_info().rss for p in child.children(recursive=True)
                )
                peak_rss = max(peak_rss, rss)
            except psutil.NoSuchProcess:
                pass
            current_available = psutil.virtual_memory().available
            lowest_available = min(lowest_available, current_available)
            if peak_rss > rss_limit or current_available < 768 * 1024**2:
                reason = "stopped_measured_memory_limit"
            elapsed = time.monotonic() - started
            if elapsed >= args.wall_seconds:
                reason = "stopped_wall_limit"
            if elapsed >= next_disk_check:
                max_scratch = max(max_scratch, tree_bytes(args.scratch_root))
                max_persistent = max(max_persistent, tree_bytes(args.output))
                if max_scratch > 8 * 1024**3 or max_persistent > 4 * 1024**3:
                    reason = "stopped_storage_limit"
                next_disk_check = elapsed + 15
            if reason:
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.25)
        exit_code = process.wait()
    report.update(
        {
            "status": reason or ("completed" if exit_code == 0 else "worker_stopped_no_retry"),
            "exit_code": exit_code,
            "wall_seconds": time.monotonic() - started,
            "sampled_peak_worker_rss_bytes": peak_rss,
            "lowest_available_ram_bytes": lowest_available,
            "maximum_scratch_bytes": max_scratch,
            "maximum_persistent_bytes": max_persistent,
            "final_persistent_bytes": tree_bytes(args.output),
        }
    )
    with (args.output / "supervisor.json").open("x") as saved:
        json.dump(report, saved, indent=2)
        saved.write("\n")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
