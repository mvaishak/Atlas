"""CPU preparation guards require no native Torch or FAISS on the Mac."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def adapter() -> Any:
    spec = importlib.util.spec_from_file_location("cpu_adapter", "scripts/m4_colab_cpu.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_proposal_denied_before_inputs_or_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "proposal.json"
    manifest.write_text(json.dumps({"execution_approved": False}))
    with pytest.raises(PermissionError):
        adapter().execution_spec(manifest, tmp_path / "absent")
    assert list(tmp_path.iterdir()) == [manifest]


def test_cpu_export_rejects_changed_or_extra_files(tmp_path: Path) -> None:
    item = tmp_path / "development.parquet"
    item.write_bytes(b"fixture-only")
    transfer = tmp_path / "CPU_TRANSFER.json"
    transfer.write_text(
        json.dumps({"files": {item.name: hashlib.sha256(item.read_bytes()).hexdigest()}})
    )
    pin = hashlib.sha256(transfer.read_bytes()).hexdigest()
    adapter().verify_export(tmp_path, pin)
    item.write_bytes(b"changed")
    with pytest.raises(ValueError):
        adapter().verify_export(tmp_path, pin)
    item.write_bytes(b"fixture-only")
    (tmp_path / "unexpected").write_bytes(b"extra")
    with pytest.raises(ValueError):
        adapter().verify_export(tmp_path, pin)


def test_cpu_export_rejects_traversal(tmp_path: Path) -> None:
    transfer = tmp_path / "CPU_TRANSFER.json"
    transfer.write_text(json.dumps({"files": {"../outside": "0" * 64}}))
    pin = hashlib.sha256(transfer.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        adapter().verify_export(tmp_path, pin)


def test_cpu_export_rejects_unmanifested_cache_payload(tmp_path: Path) -> None:
    transfer = tmp_path / "CPU_TRANSFER.json"
    transfer.write_text(json.dumps({"files": {}}))
    pin = hashlib.sha256(transfer.read_bytes()).hexdigest()
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "unexpected.parquet").write_bytes(b"unapproved")
    with pytest.raises(ValueError):
        adapter().verify_export(tmp_path, pin)


def test_cpu_platform_guard_precedes_native_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    module = adapter()
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    original = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"torch", "faiss", "numpy", "atlasrank"}:
            raise AssertionError("Native or package imports must follow the platform guard")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="CPU runtime"):
        module.runtime_smoke(tmp_path)
    with pytest.raises(ValueError, match="CPU runtime"):
        module.preflight(tmp_path, "unused", tmp_path / "absent")
