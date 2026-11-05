"""M5 cumulative allocation ledger built on the proven Colab receipt contract."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from atlasrank.retrieval.colab_runner import (
    allocation_delay,
    charged_seconds,
    full_allocation_start,
)
from atlasrank.retrieval.colab_transport import once, sha
from atlasrank.training.configuration import parse_manifest


@dataclass(frozen=True)
class AllocationSession:
    directory: Path
    receipt_path: Path
    session_id: str
    allocation_started_utc: float
    session_started_monotonic: float
    allocation_elapsed_before_start: float
    remaining_seconds: float
    manifest_sha256: str
    active_path: Path


def begin_session(
    manifest_path: Path,
    persistent: Path,
    *,
    allocation_started_utc: float,
    recover_interrupted: bool = False,
) -> AllocationSession:
    manifest = parse_manifest(json.loads(manifest_path.read_text()))
    if manifest.get("execution_approved") is not True:
        raise PermissionError("M5 pilot execution manifest is not approved")
    expected_directory = manifest.get("persistent_destination")
    token = manifest.get("operator_owner_token")
    if not expected_directory or persistent.resolve() != Path(expected_directory).resolve():
        raise PermissionError("M5 persistent destination differs from the approved manifest")
    drive_root = Path("/content/drive/MyDrive/AtlasRank/m5").resolve()
    if not persistent.resolve().is_relative_to(drive_root):
        raise PermissionError("M5 checkpoints must use the approved private Drive subtree")
    if not token or os.environ.get("ATLAS_COLAB_EXCLUSIVE_RUNTIME") != token:
        raise PermissionError("M5 runtime owner token is missing or differs")
    persistent.mkdir(parents=True, exist_ok=True)
    owner_path = persistent / "OWNER.json"
    owner = {
        "owner_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "manifest_sha256": sha(manifest_path),
    }
    if owner_path.exists():
        if json.loads(owner_path.read_text()) != owner:
            raise PermissionError("M5 persistent destination belongs to a different run")
    else:
        approved_copy = persistent / "approved-m5-pilot.json"
        input_copy = persistent / "inputs" / str(manifest["input_archive_name"])
        expected_entries = {"approved-m5-pilot.json", "inputs"}
        if {p.name for p in persistent.iterdir()} != expected_entries:
            raise PermissionError("M5 destination contains unexpected files before ownership")
        if not approved_copy.is_file() or sha(approved_copy) != sha(manifest_path):
            raise PermissionError("Persisted approved manifest differs from local execution file")
        if (
            not input_copy.is_file()
            or sha(input_copy) != manifest.get("input_archive_sha256")
            or input_copy.stat().st_size != manifest.get("input_archive_bytes")
        ):
            raise PermissionError("Persisted training export differs from approved input pin")
        if {p.name for p in (persistent / "inputs").iterdir()} != {input_copy.name}:
            raise PermissionError("Unexpected files in persisted M5 inputs directory")
        once(owner_path, owner)
    anchor = full_allocation_start(persistent, allocation_started_utc)
    active_path = persistent / "ACTIVE.json"
    if active_path.exists():
        if not recover_interrupted:
            raise PermissionError(
                "An M5 runtime may still be active; verify it stopped, then request recovery"
            )
        recover_active_session(
            persistent,
            manifest,
            manifest_sha256=sha(manifest_path),
            old_runtime_stopped=recover_interrupted,
        )
    prior_charged = charged_seconds(persistent, float(manifest["gpu_allocation_seconds"]))
    setup_elapsed, _ = allocation_delay(persistent, anchor)
    remaining = remaining_allocation(manifest, prior_charged, setup_elapsed)
    prior_charged += float(manifest["prior_m5_gpu_allocation_seconds"])
    session_id = uuid.uuid4().hex
    receipt_path = persistent / f"{session_id}.start.json"
    once(
        receipt_path,
        {
            "schema_version": 1,
            "session_id": session_id,
            "utc_start": allocation_started_utc,
            "reserved_seconds": remaining,
            "setup_elapsed_seconds": setup_elapsed,
            "prior_charged_seconds": prior_charged,
            "manifest_sha256": sha(manifest_path),
        },
    )
    once(
        active_path,
        {
            "session_id": session_id,
            "manifest_sha256": sha(manifest_path),
            "owner_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "utc_start": allocation_started_utc,
        },
    )
    return AllocationSession(
        persistent,
        receipt_path,
        session_id,
        allocation_started_utc,
        time.monotonic(),
        setup_elapsed,
        remaining,
        sha(manifest_path),
        active_path,
    )


def finish_session(session: AllocationSession, status: str) -> dict[str, Any]:
    end_path = session.receipt_path.with_name(
        session.receipt_path.name.replace(".start.json", ".end.json")
    )
    elapsed = session.allocation_elapsed_before_start + (
        time.monotonic() - session.session_started_monotonic
    )
    final_status = status
    if elapsed - session.allocation_elapsed_before_start > session.remaining_seconds:
        final_status = "allocation_limit_exceeded"
    value = {
        "schema_version": 1,
        "session_id": session.session_id,
        "elapsed_seconds": elapsed,
        "status": final_status,
        "utc_end": time.time(),
        "manifest_sha256": session.manifest_sha256,
    }
    once(end_path, value)
    active = json.loads(session.active_path.read_text())
    if active.get("session_id") != session.session_id:
        raise PermissionError("M5 active-session lock differs from the finishing run")
    once(
        session.directory / f"{session.session_id}.lock-release.json",
        {
            "session_id": session.session_id,
            "utc_released": time.time(),
            "status": final_status,
        },
    )
    session.active_path.unlink()
    return value


def recover_active_session(
    store: Path,
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    old_runtime_stopped: bool,
) -> dict[str, Any]:
    """Release a stale active marker only after caller confirms the old VM is stopped."""
    active_path = store / "ACTIVE.json"
    if not active_path.exists():
        raise FileNotFoundError("No interrupted M5 session lock exists")
    if not old_runtime_stopped:
        raise PermissionError("Operator must confirm the prior Colab runtime has stopped")
    active = json.loads(active_path.read_text())
    session_id = active.get("session_id")
    if not isinstance(session_id, str) or len(session_id) != 32:
        raise ValueError("Invalid M5 active-session marker")
    if active.get("manifest_sha256") != manifest_sha256:
        raise PermissionError("Active session belongs to a different approved manifest")
    expected_owner_hash = hashlib.sha256(
        str(manifest.get("operator_owner_token", "")).encode()
    ).hexdigest()
    if active.get("owner_token_sha256") != expected_owner_hash:
        raise PermissionError("Active session belongs to a different exclusive owner")
    start_path = store / f"{session_id}.start.json"
    end_path = store / f"{session_id}.end.json"
    if not start_path.is_file():
        raise ValueError("Active session has no immutable start receipt")
    recovery_copy = store / f"{session_id}.interrupted.json"
    if not end_path.exists():
        once(
            recovery_copy,
            {
                "session_id": session_id,
                "active_marker": active,
                "recovered_utc": time.time(),
                "policy": "Old runtime stop must be confirmed by operator before this call",
            },
        )
    else:
        once(
            recovery_copy,
            {
                "session_id": session_id,
                "active_marker": active,
                "recovered_utc": time.time(),
                "policy": "End receipt existed; stale active marker released",
            },
        )
    active_path.unlink()
    charged_seconds(store, float(manifest["gpu_allocation_seconds"]))
    return cast(dict[str, Any], json.loads(recovery_copy.read_text()))


def remaining_allocation(
    manifest: dict[str, Any], this_store_charged: float, setup: float
) -> float:
    """Carry previous-run charges once, and reserve finalization within the new cap."""
    previous = float(manifest["prior_m5_gpu_allocation_seconds"])
    remaining = float(manifest["gpu_allocation_seconds"]) - previous - this_store_charged - setup
    scope = manifest.get("execution_scope")
    if scope is not None:
        remaining = min(
            remaining, float(scope["max_new_allocation_seconds"]) - this_store_charged - setup
        )
        if remaining < 600:
            raise TimeoutError("Diagnostic needs 300 active seconds plus 300 finalization seconds")
    if remaining <= 0:
        raise TimeoutError("M5 cumulative GPU allocation budget is exhausted")
    return remaining
