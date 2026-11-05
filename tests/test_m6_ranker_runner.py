"""Job admission must stop before native execution when the environment cannot fit."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest


def arguments(tmp_path: Path) -> list[str]:
    return [
        "runner",
        "--inputs",
        str(tmp_path / "inputs"),
        "--source",
        str(tmp_path / "source"),
        "--output",
        str(tmp_path / "output"),
        "--scratch-root",
        str(tmp_path),
    ]


def test_mac_is_refused_before_any_native_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    before = "torch" in sys.modules
    with pytest.raises(SystemExit) as result:
        runpy.run_path("scripts/run_m6_ranker.py", run_name="__main__")
    assert result.value.code == 2
    assert not (tmp_path / "output").exists()
    assert ("torch" in sys.modules) == before


def test_low_available_ram_stops_before_native_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=7 * 1024**3))
    with pytest.raises(SystemExit) as result:
        runpy.run_path("scripts/run_m6_ranker.py", run_name="__main__")
    assert result.value.code == 2
    evidence = json.loads((tmp_path / "output/supervisor.json").read_text())
    assert evidence["native_process_started"] is False
    assert "insufficient_measured_start_ram" in evidence["status"]


def test_prior_fit_directory_cannot_be_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "FIT_STARTED.json"
    marker.write_text("preserve attempted fit")
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(SystemExit) as result:
        runpy.run_path("scripts/run_m6_ranker.py", run_name="__main__")
    assert result.value.code == 2
    assert marker.read_text() == "preserve attempted fit"


def test_short_wall_budget_refuses_native_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + ["--wall-seconds", "30"])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=10 * 1024**3))
    with pytest.raises(SystemExit) as result:
        runpy.run_path("scripts/run_m6_ranker.py", run_name="__main__")
    assert result.value.code == 2
    evidence = json.loads((tmp_path / "output/supervisor.json").read_text())
    assert evidence["status"] == "insufficient_wall_budget_for_native_gate_and_real_workload"
    assert evidence["native_process_started"] is False
