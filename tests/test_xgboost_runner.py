"""Platform refusal must precede any native XGBoost import or job execution."""

import importlib.metadata
import json
import runpy
import sys
from pathlib import Path

import pytest


def test_runner_refuses_mac_and_preserves_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "evidence"
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "argv", ["check_xgboost_boundary.py", "--output", str(destination)])
    before = "xgboost" in sys.modules
    with pytest.raises(SystemExit) as exc:
        runpy.run_path("scripts/check_xgboost_boundary.py", run_name="__main__")
    assert exc.value.code == 2
    result = json.loads((destination / "validation.json").read_text())
    assert result["status"] == "blocked_platform"
    assert result["native_execution_started"] is False
    assert ("xgboost" in sys.modules) == before


def test_runner_refuses_to_overwrite_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["check_xgboost_boundary.py", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path("scripts/check_xgboost_boundary.py", run_name="__main__")
    assert exc.value.code == 2
    assert not (tmp_path / "validation.json").exists()


def test_linux_preflight_rejects_wrong_versions_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    destination = tmp_path / "incompatible"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "version_info", (3, 12))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "not-the-reviewed-version")
    monkeypatch.setattr(sys, "argv", ["check_xgboost_boundary.py", "--output", str(destination)])

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Version mismatch must stop before starting a native worker")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path("scripts/check_xgboost_boundary.py", run_name="__main__")
    assert exc.value.code == 2
    result = json.loads((destination / "validation.json").read_text())
    assert result["status"] == "blocked_dependencies"
    assert result["native_execution_started"] is False
