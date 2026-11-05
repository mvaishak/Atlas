"""The M8 controller preserves its explicit resource and workload contract."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def helper(monkeypatch: Any) -> Any:
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("m8_controller", "scripts/run_m8_benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m8_execution_proposal_and_expanded_caps_fail_closed(monkeypatch: Any) -> None:
    import json

    module = helper(monkeypatch)
    config = json.loads(Path("configs/m8-benchmark.proposed.json").read_text())
    with pytest.raises(ValueError, match="approval"):
        module.validate_execution(config)
    config.update(approved=True, execution_deadline_utc=1.0)
    module.validate_execution(config)
    for field, value in (
        ("wall_seconds", 14401),
        ("new_storage_bytes", 2 * 1024**3),
        ("rss_bytes", 13 * 1024**3),
        ("minimum_available_bytes", 7 * 1024**3),
        ("gpus", 1),
        ("measured_requests", 999),
        ("warmup_requests", 99),
        ("execution_deadline_utc", float("nan")),
        ("execution_deadline_utc", float("inf")),
    ):
        with pytest.raises(ValueError):
            module.validate_execution({**config, field: value})


def test_storage_budget_includes_failed_and_active_outputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    native_before = {name for name in ("torch", "xgboost") if name in sys.modules}
    module = helper(monkeypatch)
    (tmp_path / "failed").mkdir()
    (tmp_path / "failed/partial.jsonl").write_bytes(b"x" * 13)
    (tmp_path / "current.jsonl").write_bytes(b"y" * 17)
    assert module.storage_bytes(tmp_path) == 30
    with (tmp_path / "current.jsonl").open("ab") as stream:
        stream.write(b"more")
    assert module.storage_bytes(tmp_path) == 34
    assert {name for name in ("torch", "xgboost") if name in sys.modules} == native_before


def test_source_snapshot_is_bound_to_registered_manifest(tmp_path: Path, monkeypatch: Any) -> None:
    import hashlib
    import json

    module = helper(monkeypatch)
    source = tmp_path / "source.py"
    source.write_text("original")
    manifest = tmp_path / "snapshot-manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "source.py", "sha256": module.digest_file(source)}]})
    )
    registered = hashlib.sha256(manifest.read_bytes()).hexdigest()
    module.verify_source_snapshot(tmp_path, registered)
    source.write_text("changed")
    manifest.write_text(
        json.dumps({"files": [{"path": "source.py", "sha256": module.digest_file(source)}]})
    )
    with pytest.raises(ValueError, match="Registered source manifest"):
        module.verify_source_snapshot(tmp_path, registered)
