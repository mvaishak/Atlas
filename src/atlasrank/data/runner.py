"""Run acquisition/preparation in a monitored local subprocess, preserving failures."""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from atlasrank.data.common import MEMORY_LIMIT, check_disk, write_json


def run_bounded(task: str, config: Path, root: Path, output: Path | None = None) -> None:
    if task not in {"acquire", "prepare", "bm25-build", "bm25-evaluate", "dense-calibrate"}:
        raise ValueError("Unsupported data task")
    if task == "dense-calibrate" and psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("CPU pilot requires at least 4 GiB available system RAM")
    check_disk(root)
    if output is not None and not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Prepared output must stay within the budgeted data root")
    command = [sys.executable, "-m", "atlasrank.data.worker", task, str(config), str(root)]
    if output is not None:
        command.append(str(output))
    environment = os.environ.copy()
    environment.update(POLARS_MAX_THREADS="2", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if task == "dense-calibrate":
        environment.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    records = root / "execution"
    records.mkdir(exist_ok=True)
    record = records / f"{task}-{uuid.uuid4().hex}.json"
    peak = 0
    started = time.monotonic()
    status = "failed"
    detail = ""
    child = subprocess.Popen(command, env=environment)
    try:
        process = psutil.Process(child.pid)
        last_disk = 0.0
        while child.poll() is None:
            if task == "dense-calibrate" and time.monotonic() - started >= 600:
                raise RuntimeError(
                    "10-minute CPU pilot command cap exceeded; partial files preserved"
                )
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                break
            peak = max(peak, rss)
            if rss > MEMORY_LIMIT:
                raise RuntimeError("8 GiB RSS limit exceeded; partial files preserved")
            if time.monotonic() - last_disk >= 1:
                check_disk(root)
                last_disk = time.monotonic()
            time.sleep(0.1)
        code = child.wait()
        check_disk(root)
        if code:
            raise RuntimeError(f"Data subprocess failed with exit code {code}")
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
        write_json(
            record,
            {
                "schema_version": 1,
                "task": task,
                "status": status,
                "detail": detail,
                "sampled_peak_child_rss_bytes": peak,
                "rss_limit_bytes": MEMORY_LIMIT,
                "monitor_interval_seconds": 0.1,
                "elapsed_seconds": time.monotonic() - started,
                "command": command,
            },
        )
        print(f"Execution record: {record}", flush=True)
