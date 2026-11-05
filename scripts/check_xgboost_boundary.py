"""Bounded, no-install Linux CPU check. Refuses macOS before native imports."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TIME_LIMIT_SECONDS = 180
RSS_LIMIT_BYTES = 2 * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output must be new; preserve previous evidence")
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "preflight",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "native_execution_started": False,
        "environment": {
            "platform": sys.platform,
            "python": sys.version,
            "executable": sys.executable,
            "machine": platform.machine(),
            "kernel": platform.release(),
            "libc": platform.libc_ver(),
        },
        "limits": {"wall_seconds": TIME_LIMIT_SECONDS, "worker_rss_bytes": RSS_LIMIT_BYTES},
    }

    def finish(status: str, code: int) -> int:
        result["status"] = status
        result["exit_code"] = code
        (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(status, output)
        return code

    if sys.platform != "linux":
        return finish("blocked_platform", 2)
    if sys.version_info[:2] != (3, 12):
        return finish("blocked_python_version", 2)
    requirements = [
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
    ]
    spec = tomllib.loads((repo / "pyproject.toml").read_text())
    pins = {}
    for requirement in spec["project"]["dependencies"] + spec["dependency-groups"]["dev"]:
        name, version = requirement.split("==", 1)
        if name in requirements:
            pins[name] = version.split(";", 1)[0].strip()
    versions = {}
    for name in [*requirements, "pydantic", "starlette"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result["environment"]["packages"] = versions
    result["required_pins"] = pins
    if set(pins) != set(requirements) or any(versions[n] != pins[n] for n in requirements):
        return finish("blocked_dependencies", 2)
    source_paths = [
        "pyproject.toml",
        "uv.lock",
        "configs/ranking-features-v1.json",
        "src/atlasrank/features/ranking.py",
        "src/atlasrank/ranking/prediction.py",
        "src/atlasrank/ranking/xgboost_boundary.py",
        "src/atlasrank/evaluation/ranking.py",
        "src/atlasrank/serving/ranking.py",
        "src/atlasrank/serving/app.py",
        "src/atlasrank/retrieval/bm25.py",
        "src/atlasrank/retrieval/analyzer.py",
        "src/atlasrank/retrieval/dense.py",
        "scripts/check_xgboost_boundary.py",
        "tests/test_ranking_features.py",
        "tests/test_ranking_parity.py",
        "tests/test_xgboost_capture.py",
        "tests/test_xgboost_boundary.py",
    ]
    result["source_sha256"] = {
        p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in source_paths
    }
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "pytest",
        "-q",
        "tests/test_xgboost_boundary.py",
        "tests/test_xgboost_capture.py",
        "tests/test_ranking_features.py",
        "tests/test_ranking_parity.py",
        f"--junitxml={output / 'junit.xml'}",
        f"--basetemp={output / 'temporary-fixtures'}",
    ]
    result["command"] = command
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    env.update(
        {
            "ATLASRANK_XGB_EVIDENCE_DIR": str(output),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(repo / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    start = time.monotonic()
    peak_rss = 0
    stop = None
    result["native_execution_started"] = True
    # No installation, downloads, remote calls or Mac dependency changes.
    with (output / "pytest.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        while process.poll() is None:
            status = Path(f"/proc/{process.pid}/status")
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        peak_rss = max(peak_rss, int(line.split()[1]) * 1024)
            except FileNotFoundError:
                pass
            if time.monotonic() - start > TIME_LIMIT_SECONDS:
                stop = "stopped_wall_limit"
            if peak_rss > RSS_LIMIT_BYTES:
                stop = "stopped_rss_limit"
            if stop:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.1)
        returncode = process.wait()
    result.update(
        {
            "child_exit_code": returncode,
            "wall_seconds": time.monotonic() - start,
            "sampled_peak_worker_rss_bytes": peak_rss,
        }
    )
    result["evidence_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()
    }
    if stop:
        return finish(stop, 1)
    if returncode:
        return finish("native_or_test_failure", 1)
    if not (output / "native-boundary.json").is_file():
        return finish("missing_native_evidence", 1)
    try:
        suites = ET.parse(output / "junit.xml").getroot().iter("testsuite")
    except (OSError, ET.ParseError):
        return finish("missing_or_invalid_junit", 1)
    counts = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for name in counts:
            counts[name] += int(suite.get(name, "0"))
    result["test_counts"] = counts
    if counts["tests"] < 42 or any(counts[n] for n in ("failures", "errors", "skipped")):
        return finish("incomplete_native_checks", 1)
    return finish("passed", 0)


if __name__ == "__main__":
    raise SystemExit(main())
