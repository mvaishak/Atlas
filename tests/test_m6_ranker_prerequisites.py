"""Byte checks must fail closed without importing the native ML stack."""

import json
import runpy
from pathlib import Path
from typing import Any

import pytest


def checker() -> Any:
    return runpy.run_path("scripts/check_m6_ranker_prerequisites.py")["verify_file"]


def test_changed_or_missing_bytes_cannot_pass(tmp_path: Path) -> None:
    verify = checker()
    source = tmp_path / "weights"
    source.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert verify(source, expected)["status"] == "passed"
    source.write_bytes(b"abd")
    assert verify(source, expected)["status"] == "checksum_mismatch"
    assert verify(tmp_path / "missing", expected)["status"] == "missing"


def test_symlink_cannot_substitute_for_pinned_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"abc")
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    assert checker()(alias, "unused")["status"] == "symlink_rejected"


def test_inventory_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    marker = tmp_path / "validation.json"
    marker.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["checker", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path("scripts/check_m6_ranker_prerequisites.py", run_name="__main__")
    assert result.value.code == 2
    assert marker.read_text() == "preserve"


def test_absent_input_tree_preserves_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    destination = tmp_path / "evidence"
    monkeypatch.setattr(
        sys, "argv", ["checker", "--root", str(tmp_path), "--output", str(destination)]
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path("scripts/check_m6_ranker_prerequisites.py", run_name="__main__")
    assert result.value.code == 2
    evidence = json.loads((destination / "validation.json").read_text())
    assert evidence["errors"]
    assert evidence["byte_inventory_passed"] is False
    assert evidence["fit_executed"] is False
