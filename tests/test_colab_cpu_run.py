"""Pure control-plane checks; never import native ML libraries on the Mac."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def runner() -> Any:
    spec = importlib.util.spec_from_file_location("cpu_run", "scripts/run_m4_colab_cpu.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_run_rejects_proposal_before_platform_or_work(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        runner().approved_spec(Path("configs/m4-colab-cpu-evaluation.proposed.json"))
    assert not list(tmp_path.iterdir())


def test_cpu_budget_includes_preflight_and_retries() -> None:
    assert runner().remaining_seconds(0) == 6600
    assert runner().remaining_seconds(17) == 6583
    with pytest.raises(RuntimeError):
        runner().remaining_seconds(6600)
    with pytest.raises(ValueError):
        runner().remaining_seconds(-1)


@pytest.mark.parametrize("field,value", [("rss_limit_bytes", 16 * 1024**3), ("threads", 8)])
def test_cpu_run_rejects_limit_or_recipe_drift(tmp_path: Path, field: str, value: int) -> None:
    spec = json.loads(Path("configs/m4-colab-cpu-evaluation.proposed.json").read_text())
    spec.update(
        execution_approved=True,
        runnable=True,
        destination_confirmed=True,
        persistent_destination=spec["proposed_persistent_destination"],
        operator_owner_token="fixture-owner",
    )
    spec[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        runner().approved_spec(path)
