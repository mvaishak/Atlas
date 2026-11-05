"""Thin CPU-only operator adapter; all retrieval/evaluation logic stays in the package."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_export(root: Path, expected: str) -> dict[str, Any]:
    transfer = root / "CPU_TRANSFER.json"
    if sha(transfer) != expected:
        raise ValueError("CPU transfer manifest differs")
    spec: dict[str, Any] = json.loads(transfer.read_text())
    for name, pin in spec["files"].items():
        p = root / name
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or sha(p) != pin:
            raise ValueError("CPU export file differs or escapes root")
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("CPU export symlink forbidden")
        if p.is_file():
            actual.add(str(p.relative_to(root)))
    if actual != set(spec["files"]) | {"CPU_TRANSFER.json"}:
        raise ValueError("Unexpected CPU export inventory")
    return spec


def execution_spec(path: Path, root: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text())
    if spec.get("execution_approved") is not True or spec.get("runnable") is not True:
        raise PermissionError("Full CPU evaluation requires explicit manifest approval")
    verify_export(root, spec["transfer_manifest_sha256"])
    return spec


def platform_guard() -> None:
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or platform.python_version() != "3.13.18"
        or os.environ.get("KMP_DUPLICATE_LIB_OK")
    ):
        raise ValueError("Unreviewed CPU runtime or unsafe OpenMP override")


def runtime_smoke(directory: Path) -> dict[str, Any]:
    platform_guard()
    import faiss
    import numpy as np
    import torch

    if str(torch.__version__) != "2.11.0+cpu" or torch.cuda.is_available():
        raise ValueError("Unreviewed CPU runtime")
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    x = torch.eye(4, dtype=torch.float32)
    values = (x @ x.T).numpy()
    index = faiss.IndexFlatIP(4)
    index.add(values)
    path = directory / "smoke.faiss"
    faiss.write_index(index, str(path))
    loaded = faiss.read_index(str(path))
    np.testing.assert_array_equal(loaded.reconstruct_n(0, 4), values)
    scores, rows = loaded.search(values, 1)
    np.testing.assert_array_equal(rows[:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(scores[:, 0], [1, 1, 1, 1])
    return {
        "torch": str(torch.__version__),
        "torch_file": torch.__file__,
        "faiss": faiss.__version__,
        "save_load_exact": True,
        "cpu_only": True,
    }


def preflight(root: Path, pin: str, output: Path) -> None:
    platform_guard()
    # Run with -B/PYTHONDONTWRITEBYTECODE so exact inventory remains valid on reuse.
    sys.dont_write_bytecode = True
    import importlib.metadata as md

    from atlasrank.data.prepare import load_judgments
    from atlasrank.retrieval.colab_environment import reviewed_source_hashes
    from atlasrank.retrieval.colab_full_contract import verify_inputs
    from atlasrank.retrieval.encoder import PinnedEncoder, configuration

    started = time.monotonic()
    transfer = verify_export(root, pin)
    full = transfer["full_contract"]
    if reviewed_source_hashes() != full["source_sha256"]:
        raise ValueError("Frozen package source differs")
    verify_inputs(root / "full-input", full)
    environment = json.loads((root / "environment/environment.json").read_text())
    versions = {name: md.version(name) for name in environment["pinned_packages"]}
    if versions != environment["pinned_packages"]:
        raise ValueError("CPU overlay versions differ")
    output.mkdir(parents=True, exist_ok=False)
    runtime = runtime_smoke(output)
    development = load_judgments(root / "prepared", "development")
    if development["query_id"].n_unique() != 4129:
        raise ValueError("Approved development query coverage differs")
    checks = [
        "test_dense_infrastructure.py::test_bundle_save_load_native_candidates_and_expected_origin",
        "test_dense_infrastructure.py::test_development_job_tracks_all_candidates_unknowns_and_undefined_recall",
        "test_colab_full.py::test_receipt_restart_no_reencoding_and_coverage",
        "test_colab_full.py::test_streaming_return_native_bundle_and_separate_local_budget",
    ]
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "cache_dir=" + str(output / "pytest-cache"),
            *checks,
        ],
        cwd=root / "fixtures",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (output / "fixtures.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    cfg = configuration(root / "full-input/source/configs/dense.json")
    cfg["model_path"] = str(root / "full-input/model")
    encoder = PinnedEncoder(cfg)
    # Synthetic query probes only: never re-encode catalog products during preflight.
    probes = ["atlasrank synthetic query", "", "  WIRELESS\tHeadphones ", "café ﬃ"]
    import numpy as np

    a, b = encoder.encode(probes), encoder.encode(probes)
    np.testing.assert_array_equal(a, b)
    result = {
        "schema_version": 1,
        "scope": "bounded CPU preflight only",
        "runtime": runtime,
        "versions": versions,
        "development_sha256": sha(root / "prepared/development.parquet"),
        "development_rows": development.height,
        "development_queries": 4129,
        "fixture_command": run.args,
        "fixture_status": "passed",
        "synthetic_query_repeat_exact": True,
        "query_recipe": encoder.recipe,
        "full_evaluation_executed": False,
        "products_reencoded": 0,
        "test_judgments_accessed": False,
        "seconds": time.monotonic() - started,
    }
    (output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["verify", "smoke", "preflight"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pin", default="")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "verify":
        print(json.dumps(verify_export(a.root, a.pin), indent=2))
    elif a.action == "smoke":
        a.output.mkdir(parents=True, exist_ok=False)
        print(json.dumps(runtime_smoke(a.output), indent=2))
    else:
        preflight(a.root, a.pin, a.output)


if __name__ == "__main__":
    main()
