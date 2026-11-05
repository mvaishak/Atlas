"""Mac supervisor: account preflight failures and bound native import/evaluation."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import psutil

from atlasrank.data.common import check_disk
from atlasrank.retrieval.colab_full_contract import approve_full
from atlasrank.retrieval.colab_runner import charged_seconds, size
from atlasrank.retrieval.colab_transport import once, sha


def run(manifest: Path, inputs: Path, archive: Path, expected_sha: str, work: Path) -> None:
    entered = time.monotonic()
    utc = datetime.now(UTC).timestamp()
    flags = json.loads(manifest.read_text())
    if flags.get("local_execution_approved") is not True or flags.get("runnable") is not True:
        raise PermissionError("Full local execution requires explicit approval")
    if not work.resolve().is_relative_to(Path("data/esci").resolve()):
        raise ValueError("Local workspace must remain under budgeted data root")
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".local-supervisor.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        charged = charged_seconds(work, 7200)
        if charged >= 7200:
            raise RuntimeError("Local cumulative time exhausted")
        attempt = uuid.uuid4().hex
        once(
            work / (attempt + ".start.json"), {"utc_start": utc, "reserved_seconds": 7200 - charged}
        )
        status = "failed"
        child = None
        try:
            # Metadata-only gate here; all large hashes run in the watched child.
            approve_full(manifest, inputs, work, local=True, verify_payload=False)
            if archive.stat().st_size > 4 * 1024**3:
                raise ValueError("Return archive exceeds reservation")
            if psutil.virtual_memory().available < 6 * 1024**3:
                raise RuntimeError("Local full import requires 6 GiB available RAM")
            binding = {"manifest_sha256": sha(manifest), "return_sha256": expected_sha}
            if (work / "LOCAL_RUN.json").exists():
                if json.loads((work / "LOCAL_RUN.json").read_text()) != binding:
                    raise ValueError("Local resume source/return differs")
            else:
                once(work / "LOCAL_RUN.json", binding)
            account = work / "local-budget"
            account.mkdir(exist_ok=True)
            (account / "attempts").mkdir(exist_ok=True)
            if not (account / "JOB.json").exists():
                once(account / "JOB.json", {"wall_seconds": 7200})
            deadline = entered + 7200 - charged
            env = dict(
                os.environ,
                ATLAS_MAC_PARENT_PID=str(os.getpid()),
                ATLAS_MAC_DEADLINE=str(deadline),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                POLARS_MAX_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                HF_HUB_OFFLINE="1",
            )
            output = work / ("attempt-" + attempt)
            command = [
                sys.executable,
                "-m",
                "atlasrank.retrieval.colab_full_return",
                "--manifest",
                str(manifest),
                "--inputs",
                str(inputs),
                "--store",
                str(archive),
                "--return-sha256",
                expected_sha,
                "--work",
                str(work),
                "--output",
                str(output),
            ]
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("process_guard.py")),
                    "--parent",
                    str(os.getpid()),
                    "--deadline",
                    str(deadline),
                    "--rss-limit",
                    str(8 * 1024**3),
                    "--",
                    *command,
                ],
                env=env,
                start_new_session=True,
            )
            process = psutil.Process(child.pid)
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Two-hour local cap exceeded")
                try:
                    rss = max(
                        p.memory_info().rss for p in [process, *process.children(recursive=True)]
                    )
                except psutil.NoSuchProcess:
                    continue
                if (
                    size(work) + archive.stat().st_size > 8 * 1024**3
                    or rss > 8 * 1024**3
                    or psutil.virtual_memory().available < 1024**3
                ):
                    raise RuntimeError("Local resource ceiling exceeded")
                check_disk(Path("data/esci"))
                time.sleep(0.2)
            if child.wait():
                raise RuntimeError("Local phase failed; preserve outputs")
            status = "passed"
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            once(
                work / (attempt + ".end.json"),
                {
                    "elapsed_seconds": time.monotonic() - entered,
                    "status": status,
                    "utc_end": time.time(),
                },
            )


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("manifest", "inputs", "archive", "work"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--return-sha256", required=True)
    a = p.parse_args()
    run(a.manifest, a.inputs, a.archive, a.return_sha256, a.work)


if __name__ == "__main__":
    main()
