"""Approval-gated Colab bootstrap and cumulative allocation supervisor."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlasrank.retrieval.colab_environment import ROOT, approval, validate_runtime
from atlasrank.retrieval.colab_transport import once, sha


def size(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def charged_seconds(store: Path, cap: float = 600) -> float:
    """Unclosed sessions conservatively charge downtime; budget never resets."""
    total = 0.0
    for start in sorted(store.glob("*.start.json")):
        entry = json.loads(start.read_text())
        end = start.with_name(start.name.replace(".start.json", ".end.json"))
        recovery = start.with_name(start.name.replace(".start.json", ".recovered.json"))
        if end.exists():
            seconds = json.loads(end.read_text())["elapsed_seconds"]
        elif recovery.exists():
            seconds = json.loads(recovery.read_text())["charged_seconds"]
        else:
            elapsed = datetime.now(UTC).timestamp() - entry["utc_start"]
            seconds = (
                entry["reserved_seconds"]
                if elapsed < 0
                else min(entry["reserved_seconds"], elapsed)
            )
            once(
                recovery,
                {
                    "charged_seconds": seconds,
                    "includes_downtime": True,
                    "charged_through_utc": entry["utc_start"] + seconds,
                },
            )
        if not isinstance(seconds, (int, float)) or not 0 <= seconds <= cap * 2:
            raise ValueError("Invalid session charge")
        total += seconds
    return total


def allocation_delay(
    store: Path, allocated: float, *, now: float | None = None
) -> tuple[float, float]:
    """Charge connection/setup time once, including gaps between managed sessions."""
    current = time.time() if now is None else now
    if not 0 < allocated <= current:
        raise ValueError("Allocation timestamp is in the future or invalid")
    accounted = allocated
    for end in store.glob("*.end.json"):
        accounted = max(accounted, float(json.loads(end.read_text()).get("utc_end", allocated)))
    for recovery in store.glob("*.recovered.json"):
        accounted = max(
            accounted, float(json.loads(recovery.read_text()).get("charged_through_utc", allocated))
        )
    if accounted > current:
        raise ValueError("Allocation ledger clock is in the future")
    return current - accounted, accounted


def full_allocation_start(store: Path, supplied: float) -> float:
    """Persist the first full allocation; charge all gaps, even disconnected time."""
    if not 0 < supplied <= time.time():
        raise ValueError("Invalid actual runtime allocation timestamp")
    anchor = store / "FIRST_ALLOCATION.json"
    if not anchor.exists():
        once(anchor, {"utc_start": supplied, "policy": "All gaps charged conservatively"})
    first = float(json.loads(anchor.read_text())["utc_start"])
    if first > supplied or first <= 0:
        raise ValueError("Full allocation anchor differs; stop for clock/accounting review")
    return first


def bounded_command(
    command: list[str],
    log: Path,
    deadline: float,
    spec: dict[str, Any],
    persistent: Path,
    work: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    if started >= deadline:
        raise RuntimeError("Cumulative pilot time exhausted before subprocess")
    next_storage_check = 0.0
    with log.open("xb") as output:
        guarded = [
            sys.executable,
            str(Path(__file__).with_name("process_guard.py")),
            "--parent",
            str(os.getpid()),
            "--deadline",
            str(deadline),
            "--rss-limit",
            str(spec["rss_limit_bytes"]),
            "--",
            *command,
        ]
        child = subprocess.Popen(
            guarded,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("cumulative GPU allocation/wall cap reached")
                if time.monotonic() >= next_storage_check and (
                    size(persistent) > spec["persistent_limit_bytes"]
                    or size(work) + spec.get("scratch_input_reservation_bytes", 0)
                    > spec["scratch_limit_bytes"]
                ):
                    raise RuntimeError("Pilot storage cap reached")
                if time.monotonic() >= next_storage_check and (
                    sum(size(p) for p in work.glob("*/wheels"))
                    > spec["dependency_download_limit_bytes"]
                ):
                    raise RuntimeError("Dependency download payload cap reached")
                if time.monotonic() >= next_storage_check:
                    next_storage_check = time.monotonic() + 5
                status = Path(f"/proc/{child.pid}/status")
                if status.exists():
                    for line in status.read_text().splitlines():
                        if (
                            line.startswith("VmRSS:")
                            and int(line.split()[1]) * 1024 > spec["rss_limit_bytes"]
                        ):
                            raise RuntimeError("Pilot subprocess RSS cap reached")
                time.sleep(0.2)
            if child.wait() != 0:
                if time.monotonic() >= deadline:
                    raise TimeoutError("cumulative GPU allocation/wall cap reached")
                raise RuntimeError(f"Pilot phase failed; inspect {log}")
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    if size(persistent) > spec["persistent_limit_bytes"]:
        raise RuntimeError("Persistent storage limit exceeded at phase completion")
    return {
        "command": command,
        "elapsed_seconds": time.monotonic() - started,
        "log_sha256": sha(log),
    }


def run(manifest: Path, inputs: Path, persistent: Path, work: Path) -> None:
    spec = approval(manifest, inputs, persistent)
    execute_session(manifest, inputs, persistent, work, spec)


def execute_session(
    manifest: Path,
    inputs: Path,
    persistent: Path,
    work: Path,
    spec: dict[str, Any],
    *,
    worker_module: str = "atlasrank.retrieval.colab_pilot",
    allocation_started_utc: float | None = None,
) -> None:
    """Internal bootstrap shared by independently gated pilot/full entry points."""
    if not persistent.parent.is_dir():
        raise RuntimeError("Approved Drive parent must already be mounted")
    persistent.mkdir(exist_ok=True)
    owner = {"owner_token": spec["operator_owner_token"], "manifest_sha256": sha(manifest)}
    if (persistent / "OWNER.json").exists():
        prior = json.loads((persistent / "OWNER.json").read_text())
        if prior != owner:
            amendment = persistent / ("OWNER-" + owner["manifest_sha256"] + ".json")
            if amendment.exists():
                if json.loads(amendment.read_text()) != owner:
                    raise PermissionError("Owner amendment differs")
            elif (
                spec.get("kind") != "m4_colab_full_catalog"
                and spec.get("previous_manifest_sha256") == prior["manifest_sha256"]
                and prior["owner_token"] == owner["owner_token"]
                and not (persistent / "chunks").exists()
                and not (persistent / "control").exists()
            ):
                # Bootstrap-only repair, before any embedding job. Keep original
                # ownership and all session charges; never amend encoded recipes.
                once(amendment, owner)
            else:
                raise PermissionError("Persistent owner/configuration differs; preserve history")
    else:
        once(persistent / "OWNER.json", owner)
    # Drive doesn't promise cross-runtime POSIX exclusion. Operator attests prior runtime stopped.
    if os.environ.get("ATLAS_COLAB_EXCLUSIVE_RUNTIME") != spec["operator_owner_token"]:
        raise PermissionError("Operator must attest exclusive runtime ownership")
    supplied_allocation = allocation_started_utc
    if spec.get("kind") == "m4_colab_full_catalog":
        if allocation_started_utc is None:
            raise PermissionError("Actual allocation timestamp required")
        allocation_started_utc = full_allocation_start(persistent, allocation_started_utc)
    charged = charged_seconds(persistent, spec["wall_seconds"]) + spec.get(
        "prior_allocation_charge_seconds", 0
    )
    pre_session = 0.0
    accounted_start = time.time()
    if allocation_started_utc is not None:
        pre_session, accounted_start = allocation_delay(persistent, allocation_started_utc)
    if charged + pre_session >= spec["wall_seconds"]:
        raise RuntimeError("Cumulative pilot budget exhausted")
    attempt = uuid.uuid4().hex
    once(
        persistent / f"{attempt}.start.json",
        {
            "utc_start": accounted_start,
            "reserved_seconds": spec["wall_seconds"] - charged,
            "policy": "Charge managed setup, warmup, failures and restarts",
        },
    )
    started = time.monotonic()
    deadline = started + spec["wall_seconds"] - charged - pre_session
    workspace = work / attempt
    workspace.mkdir(parents=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        NVIDIA_TF32_OVERRIDE="0",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        POLARS_MAX_THREADS="2",
        ATLAS_COLAB_DEADLINE=str(deadline),
        ATLAS_COLAB_PARENT_PID=str(os.getpid()),
    )
    commands: list[dict[str, Any]] = []
    status = "failed"
    detail = ""
    try:
        # Discovery happens inside the managed session; its actual environment is exported.
        discovery = workspace / "discovery.json"
        commands.append(
            bounded_command(
                [sys.executable, "-m", "atlasrank.retrieval.colab_environment", "inspect"],
                discovery,
                deadline,
                spec,
                persistent,
                work,
                env,
            )
        )
        info = json.loads(discovery.read_text())
        once(persistent / f"{attempt}.discovery.json", info)
        environment = json.loads((ROOT / "environments/colab/environment.json").read_text())
        validate_runtime(info, environment)
        venv = workspace / "venv"
        commands.append(
            bounded_command(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    "--without-pip",
                    "--system-site-packages",
                    str(venv),
                ],
                workspace / "venv.log",
                deadline,
                spec,
                persistent,
                work,
                env,
            )
        )
        python = str(venv / "bin/python")
        wheelhouse = workspace / "wheels"
        commands.append(
            bounded_command(
                [
                    python,
                    "-m",
                    "pip",
                    "download",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "--no-deps",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--dest",
                    str(wheelhouse),
                    "-r",
                    str(ROOT / "environments/colab/requirements.txt"),
                ],
                workspace / "download.log",
                deadline,
                spec,
                persistent,
                work,
                env,
            )
        )
        if size(wheelhouse) > spec["dependency_download_limit_bytes"]:
            raise RuntimeError("Dependency payload cap exceeded")
        commands.append(
            bounded_command(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "--no-deps",
                    "--require-hashes",
                    "--no-index",
                    "--find-links",
                    str(wheelhouse),
                    "-r",
                    str(ROOT / "environments/colab/requirements.txt"),
                ],
                workspace / "install.log",
                deadline,
                spec,
                persistent,
                work,
                env,
            )
        )
        env["ATLAS_COLAB_SETUP_SECONDS"] = str(time.monotonic() - started)
        env["ATLAS_COLAB_NATIVE_TORCH_FILE"] = info["torch_file"]
        env["ATLAS_COLAB_NATIVE_TORCH_VERSION"] = info["torch_version"]
        commands.append(
            bounded_command(
                [
                    python,
                    "-m",
                    worker_module,
                    "--manifest",
                    str(manifest),
                    "--inputs",
                    str(inputs),
                    "--persistent",
                    str(persistent),
                    "--work",
                    str(workspace),
                ],
                workspace / "pilot.log",
                deadline,
                spec,
                persistent,
                work,
                env,
            )
        )
        if spec.get("kind") == "m4_colab_full_catalog":
            commands.append(
                bounded_command(
                    [
                        python,
                        "-m",
                        "atlasrank.retrieval.colab_full_transfer",
                        "--store",
                        str(persistent),
                        "--output",
                        str(workspace / "full-return.zip"),
                    ],
                    workspace / "return.log",
                    deadline,
                    spec,
                    persistent,
                    work,
                    env,
                )
            )
            print((workspace / "return.log").read_text(), flush=True)
        status = "passed"
    except BaseException as error:
        detail = f"{type(error).__name__}: {error}"
        raise
    finally:
        elapsed = time.monotonic() - started + pre_session
        completed_report = workspace / "evidence/report.json"
        new_titles = (
            json.loads(completed_report.read_text())["newly_encoded_titles"]
            if status == "passed" and completed_report.exists()
            else None
        )
        once(
            persistent / f"{attempt}.end.json",
            {
                "status": status,
                "utc_end": time.time(),
                "detail": detail,
                "elapsed_seconds": elapsed,
                "gpu_allocation_upper_bound_seconds": elapsed,
                "previous_charged_seconds": charged,
                "commands": commands,
                "newly_encoded_titles": new_titles,
                "managed_session_titles_per_second": (
                    new_titles / elapsed if new_titles is not None else None
                ),
                "operator_pre_session_seconds": pre_session,
                "allocation_timestamp_supplied": supplied_allocation,
                "allocation_budget_anchor": allocation_started_utc,
            },
        )
        for log in workspace.glob("*.log"):
            once(
                persistent / f"{attempt}.{log.name}.json",
                {"text": log.read_text(errors="replace")[-100000:]},
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/content/atlasrank-work"))
    args = parser.parse_args()
    run(args.manifest, args.inputs, args.persistent, args.work)


if __name__ == "__main__":
    main()
