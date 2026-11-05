"""Exercise the installed entry point without importing optional modeling code."""

import os
import subprocess
import sys
from pathlib import Path


def test_help_has_no_storage_side_effects(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "atlasrank", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "tracking-smoke" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_unknown_command_fails_without_creating_storage(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "atlasrank", "train"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_import_does_not_import_mlflow_or_torch(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, atlasrank; "
            "assert 'mlflow' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        cwd=tmp_path,
        check=True,
        env=os.environ.copy(),
    )
