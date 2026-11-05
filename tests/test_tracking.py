"""Actual SQLite/artifact integration; no mock backend and no trained model."""

import hashlib
import json
from pathlib import Path

import pytest

from atlasrank.tracking import tracking_smoke

pytestmark = pytest.mark.tracking


def test_tracking_persists_metadata_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hostile inherited URI must not redirect this local fixture to a server.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    evidence = tracking_smoke(tmp_path / "tracking with spaces")
    exported = Path(evidence.export_path)
    assert json.loads(exported.read_text()) == {"kind": "tracking_fixture", "trained_model": False}
    assert hashlib.sha256(exported.read_bytes()).hexdigest() == evidence.artifact_sha256
    assert evidence.tracking_uri.startswith("sqlite:///")
    assert evidence.artifact_uri.startswith("file:")
    assert (exported.parent / "run.json").is_file()
    assert (tmp_path / "tracking with spaces" / "tracking.db").is_file()


def test_repeated_checks_preserve_previous_run(tmp_path: Path) -> None:
    first = tracking_smoke(tmp_path)
    second = tracking_smoke(tmp_path)
    assert first.run_id != second.run_id
    assert first.artifact_sha256 == second.artifact_sha256
    assert Path(first.export_path).is_file()
    assert Path(second.export_path).is_file()
