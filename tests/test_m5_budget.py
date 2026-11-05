from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from atlasrank.training.budget import recover_active_session


def test_recovery_preserves_orphaned_session_and_charges_elapsed_time(tmp_path: Path) -> None:
    session_id = "a" * 32
    manifest_sha = "manifest-hash-v1"
    started = time.time() - 5
    (tmp_path / "ACTIVE.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "manifest_sha256": manifest_sha,
                "owner_token_sha256": hashlib.sha256(b"owner-token").hexdigest(),
            }
        )
    )
    (tmp_path / f"{session_id}.start.json").write_text(
        json.dumps({"utc_start": started, "reserved_seconds": 60})
    )
    manifest = {"gpu_allocation_seconds": 3600, "operator_owner_token": "owner-token"}
    with pytest.raises(PermissionError, match="confirm"):
        recover_active_session(
            tmp_path,
            manifest,
            manifest_sha256=manifest_sha,
            old_runtime_stopped=False,
        )
    result = recover_active_session(
        tmp_path,
        manifest,
        manifest_sha256=manifest_sha,
        old_runtime_stopped=True,
    )
    assert result["session_id"] == session_id
    assert not (tmp_path / "ACTIVE.json").exists()
    assert (tmp_path / f"{session_id}.interrupted.json").is_file()
    recovery = json.loads((tmp_path / f"{session_id}.recovered.json").read_text())
    assert 0 < recovery["charged_seconds"] <= 60
