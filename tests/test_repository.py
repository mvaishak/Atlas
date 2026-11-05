"""Source publication checks use synthetic files and explicit candidate lists."""

import importlib.util
import json
from pathlib import Path
from typing import Any


def checker() -> Any:
    spec = importlib.util.spec_from_file_location("check_repository", "scripts/check_repository.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_repository


def test_source_configs_and_clean_notebooks_are_allowed(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[Guide](docs/guide.md) [Directory](docs/)\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.md").write_text("# Guide\n")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / ".env.example").write_text("API_KEY=replace-me\n")
    (tmp_path / "example.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "outputs": [], "execution_count": None}]})
    )
    assert (
        checker()(
            tmp_path, ["README.md", "docs/guide.md", "config.json", ".env.example", "example.ipynb"]
        )
        == []
    )


def test_private_and_generated_paths_are_rejected_without_reading(tmp_path: Path) -> None:
    files = [
        "data/catalog.json",
        "results/report.md",
        "docs/reviews/M7.validation.json",
        "weights.safetensors",
        "archive.tar.gz",
        ".env",
        "client.pem",
        "docs/superpowers/plan.md",
        "Colab Notebooks/run.ipynb",
    ]
    issues = checker()(tmp_path, files)
    assert len(issues) == len(files)
    assert all(name in "\n".join(issues) for name in files)


def test_secrets_and_executed_notebooks_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("ghp_" + "a" * 36)
    (tmp_path / "run.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "execution_count": 1, "outputs": []}]})
    )
    issues = checker()(tmp_path, ["secret.txt", "run.ipynb"])
    assert any("secret" in issue.lower() for issue in issues)
    assert any("execution" in issue.lower() for issue in issues)
    (tmp_path / "run.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "outputs": [{"output_type": "stream", "text": "private"}],
                    }
                ]
            }
        )
    )
    assert any("output" in issue.lower() for issue in checker()(tmp_path, ["run.ipynb"]))


def test_markdown_links_require_published_targets(tmp_path: Path) -> None:
    (tmp_path / "private.json").write_text("{}")
    (tmp_path / "README.md").write_text("[Private](private.json) [Site](https://example.com)\n")
    assert any("link" in issue.lower() for issue in checker()(tmp_path, ["README.md"]))


def test_unsafe_paths_symlinks_and_oversized_files_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_bytes(b"a" * (2 * 1024**2 + 1))
    (tmp_path / "link.txt").symlink_to(tmp_path / "large.txt")
    (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
    files = ["../escape.txt", "/absolute.txt", "link.txt", "linked/large.txt", "large.txt"]
    issues = checker()(tmp_path, files)
    assert len(issues) == len(files)
