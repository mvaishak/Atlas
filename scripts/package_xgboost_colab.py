"""Package source-only native boundary validation and a thin Colab CPU notebook."""

import argparse
import ast
import hashlib
import json
import subprocess
import textwrap
import tomllib
import zipfile
from pathlib import Path
from typing import Any

REQUIRED = (
    "xgboost-cpu",
    "numpy",
    "scipy",
    "faiss-cpu",
    "polars",
    "pyarrow",
    "bm25s",
    "fastapi",
    "httpx",
    "pytest",
    "psutil",
)
SOURCES = (
    "pyproject.toml",
    "uv.lock",
    "configs/ranking-features-v1.json",
    "scripts/check_xgboost_boundary.py",
    "src/atlasrank/__init__.py",
    "src/atlasrank/artifacts/__init__.py",
    "src/atlasrank/artifacts/dense.py",
    "src/atlasrank/artifacts/query_recipe.py",
    "src/atlasrank/artifacts/serving_bundle.py",
    "src/atlasrank/artifacts/lexical.py",
    "src/atlasrank/data/__init__.py",
    "src/atlasrank/data/common.py",
    "src/atlasrank/data/splits.py",
    "src/atlasrank/evaluation/__init__.py",
    "src/atlasrank/evaluation/metrics.py",
    "src/atlasrank/evaluation/ranking.py",
    "src/atlasrank/features/__init__.py",
    "src/atlasrank/features/ranking.py",
    "src/atlasrank/ranking/__init__.py",
    "src/atlasrank/ranking/prediction.py",
    "src/atlasrank/ranking/xgboost_boundary.py",
    "src/atlasrank/retrieval/__init__.py",
    "src/atlasrank/retrieval/analyzer.py",
    "src/atlasrank/retrieval/bm25.py",
    "src/atlasrank/retrieval/dense.py",
    "src/atlasrank/retrieval/encoder.py",
    "src/atlasrank/retrieval/chunks.py",
    "src/atlasrank/serving/__init__.py",
    "src/atlasrank/serving/app.py",
    "src/atlasrank/serving/runtime.py",
    "src/atlasrank/serving/metrics.py",
    "src/atlasrank/retrieval/dense_bundle.py",
    "src/atlasrank/retrieval/fusion.py",
    "src/atlasrank/ranking/learned.py",
    "src/atlasrank/ranking/experiment_contract.py",
    "src/atlasrank/serving/comparison.py",
    "src/atlasrank/serving/ranking.py",
    "src/atlasrank/serving/static/demo.html",
    "src/atlasrank/serving/static/tokens.css",
    "tests/test_xgboost_boundary.py",
    "tests/test_xgboost_capture.py",
    "tests/test_ranking_features.py",
    "tests/test_ranking_parity.py",
)


def build_archive(repo: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve existing input archives")
    contents = {}
    for name in SOURCES:
        source = repo / name
        if source.is_symlink() or source.stat().st_size > 2 * 1024**2:
            raise ValueError(f"Unexpected source file: {name}")
        contents[name] = source.read_bytes()
    for name, content in contents.items():
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(content)):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                for module in modules:
                    if module.startswith("atlasrank."):
                        path = "src/" + module.replace(".", "/")
                        if path + ".py" not in contents and path + "/__init__.py" not in contents:
                            raise ValueError(f"Missing source dependency: {module}")
                continue
            else:
                continue
            if module.startswith("atlasrank."):
                path = "src/" + module.replace(".", "/")
                if path + ".py" not in contents and path + "/__init__.py" not in contents:
                    raise ValueError(f"Missing source dependency: {module}")
    spec = tomllib.loads(contents["pyproject.toml"].decode())
    pins = {}
    for requirement in spec["project"]["dependencies"] + spec["dependency-groups"]["dev"]:
        name, version = requirement.split("==", 1)
        if name in REQUIRED:
            pins[name] = version.split(";", 1)[0].strip()
    if set(pins) != set(REQUIRED):
        raise ValueError("Required test pins missing")
    lock = tomllib.loads(contents["uv.lock"].decode())
    packages: dict[str, list[dict[str, Any]]] = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], []).append(package)
    constraints: dict[str, str] = {}
    todo = list(REQUIRED)
    while todo:
        name = todo.pop()
        if name in constraints:
            continue
        matches = packages[name]
        if len({m["version"] for m in matches}) != 1:
            raise ValueError(f"Ambiguous lock version: {name}")
        constraints[name] = matches[0]["version"]
        for match in matches:
            todo.extend(d["name"] for d in match.get("dependencies", []))
    if any(constraints[n] != v for n, v in pins.items()):
        raise ValueError("Working pyproject and lock pins differ")
    contents["requirements-test.txt"] = "".join(f"{n}=={pins[n]}\n" for n in sorted(pins)).encode()
    contents["constraints-test.txt"] = "".join(
        f"{n}=={v}\n" for n, v in sorted(constraints.items())
    ).encode()
    manifest = {
        "schema_version": 1,
        "purpose": "M6 zero-update synthetic Linux CPU boundary validation only",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "snapshot": "allowlisted source bytes; commit names the base, hashes identify content",
        "required_pins": pins,
        "locked_constraints": constraints,
        "files": {
            n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
            for n, b in sorted(contents.items())
        },
    }
    contents["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if sum(len(b) for b in contents.values()) > 8 * 1024**2:
        raise ValueError("Source transfer exceeds 8 MiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return manifest


def notebook(archive_sha256: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    def add(kind: str, source: str) -> None:
        cell: dict[str, Any] = {
            "cell_type": kind,
            "id": f"m6-xgb-{len(cells)}",
            "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n",
        }
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)

    add(
        "markdown",
        """
        # AtlasRank M6 — zero-update XGBoost boundary, CPU only
        Runtime → Change runtime type → Python 3, **CPU / no accelerator**.
        Run these five code cells in order. Upload only the small source ZIP when asked.
        No Drive mount, real dataset, encoder, training or GPU is used.
        If any stage fails, run the final download cell to preserve its evidence.
        Passing local guards is not a native-gate result; inspect the runner's validation.json.
    """,
    )
    add(
        "code",
        """
        import hashlib
        import json
        import platform
        import shutil
        import stat
        import subprocess
        import sys
        import zipfile
        from pathlib import Path, PurePosixPath

        assert sys.platform == 'linux' and sys.version_info[:2] == (3, 12), sys.version
        gpu = shutil.which('nvidia-smi')
        if gpu:
            probe = subprocess.run([gpu, '-L'], capture_output=True, text=True, timeout=10)
            assert not (probe.returncode == 0 and probe.stdout.strip()), 'Select a CPU runtime'
        BASE = Path('/content/atlasrank-m6-cpu-v1')
        BASE.mkdir(exist_ok=False)
        EVIDENCE = BASE / 'evidence'
        EVIDENCE.mkdir()
        discovery = {'python': sys.version, 'platform': platform.platform(),
                     'machine': platform.machine(), 'accelerator': 'none'}
        (EVIDENCE / 'discovery.json').write_text(json.dumps(discovery, indent=2))
        print(discovery)
    """,
    )
    add(
        "code",
        """
        from google.colab import files

        uploaded = files.upload()
        assert len(uploaded) == 1, 'Upload only the AtlasRank source ZIP'
        blob = next(iter(uploaded.values()))
        assert len(blob) < 4 * 1024**2
        EXPECTED_SHA256 = '__ARCHIVE_SHA__'
        assert hashlib.sha256(blob).hexdigest() == EXPECTED_SHA256, 'Wrong archive hash'
        archive_path = BASE / 'source.zip'
        archive_path.write_bytes(blob)
        SOURCE = BASE / 'source'
        SOURCE.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            assert len(archive.namelist()) == len(set(archive.namelist()))
            assert set(archive.namelist()) == set(manifest['files']) | {'manifest.json'}
            assert sum(i.file_size for i in archive.infolist()) < 8 * 1024**2
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                assert not path.is_absolute() and '..' not in path.parts
                assert not stat.S_ISLNK(item.external_attr >> 16)
                content = archive.read(item)
                if item.filename != 'manifest.json':
                    pin = manifest['files'][item.filename]
                    assert len(content) == pin['bytes']
                    assert hashlib.sha256(content).hexdigest() == pin['sha256']
                target = SOURCE.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        (EVIDENCE / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
        (EVIDENCE / 'input-archive.sha256').write_text(EXPECTED_SHA256 + '\\n')
        print('Verified', len(manifest['files']), 'source/config files; no data/model artifacts')
    """.replace("__ARCHIVE_SHA__", archive_sha256),
    )
    add(
        "code",
        """
        VENV = BASE / 'venv'
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(VENV)],
                       check=True, timeout=120)
        PY = str(VENV / 'bin/python')
        pip = [sys.executable, '-m', 'pip', '--python', PY]
        command = pip + ['install', '--disable-pip-version-check', '--no-cache-dir',
                         '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                         '-r', str(SOURCE / 'requirements-test.txt'),
                         '-c', str(SOURCE / 'constraints-test.txt'),
                         '--report', str(EVIDENCE / 'pip-install-report.json')]
        with (EVIDENCE / 'install.log').open('w') as log:
            installed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        print((EVIDENCE / 'install.log').read_text()[-8000:])
        installed.check_returncode()  # Never loosen a pin after installation failure.
        checked = subprocess.run(pip + ['check'], capture_output=True, text=True, timeout=30)
        (EVIDENCE / 'pip-check.log').write_text(checked.stdout + checked.stderr)
        checked.check_returncode()
        frozen = subprocess.run(pip + ['freeze', '--all'], capture_output=True, text=True,
                                check=True, timeout=30)
        (EVIDENCE / 'installed.txt').write_text(frozen.stdout)
        print('Isolated test environment ready; Colab native Torch was not installed or changed')
    """,
    )
    add(
        "code",
        """
        command = [PY, str(SOURCE / 'scripts/check_xgboost_boundary.py'),
                   '--output', str(EVIDENCE / 'native-results')]
        with (EVIDENCE / 'runner.log').open('w') as log:
            run = subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=210)
        print((EVIDENCE / 'runner.log').read_text())
        result_path = EVIDENCE / 'native-results/validation.json'
        result = (json.loads(result_path.read_text()) if result_path.exists()
                  else {'status': 'missing'})
        print(json.dumps(result, indent=2))
        print('Native gate passed:', run.returncode == 0 and result.get('status') == 'passed')
        # Always run the download cell, including after a failed native check.
    """,
    )
    add(
        "code",
        """
        checksum_path = EVIDENCE / 'checksums.json'
        checksums = {str(p.relative_to(EVIDENCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(EVIDENCE.rglob('*')) if p.is_file() and p != checksum_path}
        checksum_path.write_text(json.dumps(checksums, indent=2) + '\\n')
        returned = Path('/content/atlasrank-m6-boundary-results-v1.zip')
        assert not returned.exists(), 'Preserve previous return archives'
        shutil.make_archive(str(returned.with_suffix('')), 'zip', EVIDENCE)
        print('Return ZIP SHA-256:', hashlib.sha256(returned.read_bytes()).hexdigest())
        print('Return ZIP bytes:', returned.stat().st_size)
        files.download(str(returned))
    """,
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--notebook", required=True, type=Path)
    args = parser.parse_args()
    if args.notebook.exists():
        parser.error("Preserve previous notebooks; use a new path")
    manifest = build_archive(Path(__file__).resolve().parents[1], args.output)
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(json.dumps(notebook(sha), indent=1) + "\n")
    print(
        json.dumps(
            {
                "archive": str(args.output),
                "sha256": sha,
                "bytes": args.output.stat().st_size,
                "files": len(manifest["files"]),
                "notebook": str(args.notebook),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
