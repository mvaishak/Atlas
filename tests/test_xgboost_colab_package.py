"""Allowlisted source-only Colab transfer; no dependency installs or native jobs."""

import hashlib
import json
import runpy
import zipfile
from pathlib import Path

import pytest


def test_colab_archive_is_reproducible_source_only_and_hash_verified(tmp_path: Path) -> None:
    module = runpy.run_path("scripts/package_xgboost_colab.py")
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    module["build_archive"](Path.cwd(), first)
    module["build_archive"](Path.cwd(), second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert set(archive.namelist()) == set(manifest["files"]) | {"manifest.json"}
        for name, pin in manifest["files"].items():
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == pin["sha256"]
            assert len(content) == pin["bytes"]
            assert not name.startswith(("data/", ".local/", ".venv/"))
        assert "tests/test_xgboost_boundary.py" in manifest["files"]
        assert "scripts/check_xgboost_boundary.py" in manifest["files"]
        assert "src/atlasrank/serving/ranking.py" in manifest["files"]
        assert "src/atlasrank/artifacts/query_recipe.py" in manifest["files"]
        requirements = archive.read("requirements-test.txt").decode()
        assert "xgboost-cpu==3.1.1" in requirements
        assert "torch" not in requirements and "sentence-transformers" not in requirements
        assert "pydantic==" in archive.read("constraints-test.txt").decode()
        assert len(first.read_bytes()) < 1024**2
    with pytest.raises(FileExistsError):
        module["build_archive"](Path.cwd(), first)


def test_notebook_reuses_runner_and_has_no_saved_outputs(tmp_path: Path) -> None:
    module = runpy.run_path("scripts/package_xgboost_colab.py")
    notebook = module["notebook"]("a" * 64)
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "a" * 64 in code
    assert "scripts/check_xgboost_boundary.py" in code
    assert "--without-pip" in code and "--system-site-packages" not in code
    assert "files.upload()" in code and "files.download(" in code
    assert "Booster(" not in code and "DMatrix(" not in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), "<colab-cell>", "exec")
