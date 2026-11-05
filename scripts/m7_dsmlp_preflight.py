"""Metadata-only M7 planning preflight. Run on an allocated CPU pod, never login."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

RECIPE_PACKAGES = (
    "torch",
    "sentence-transformers",
    "transformers",
    "tokenizers",
    "safetensors",
    "numpy",
    "polars",
    "pyarrow",
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def effective_available(host: int, limit: int | None, used: int) -> int:
    return min(host, max(0, limit - used)) if limit is not None else host


def require_compute(hostname: str, system: str, allocated: bool) -> None:
    if "login" in hostname.casefold():
        raise ValueError("Refusing compute on a login node")
    if system != "linux" or not allocated:
        raise ValueError("An allocated Linux Kubernetes CPU container is required")


def check_recipe(
    original: dict[str, Any],
    expected: str,
    actual: dict[str, Any],
    allow_platform_difference: bool = False,
) -> None:
    if digest(original) != expected:
        raise ValueError("Original query recipe digest differs from sealed M6 pin")
    fields = ("runtime_platform", "runtime_versions", "encoder_code_sha256")
    for key in fields:
        if key == "runtime_platform" and allow_platform_difference:
            continue
        if key not in original or original[key] != actual[key]:
            raise ValueError("Original query recipe mismatch: " + key)
    if allow_platform_difference:
        from atlasrank.artifacts.query_recipe import validate_query_recipe

        candidate = {**original, **{key: actual[key] for key in fields}}
        validate_query_recipe(candidate, expected, original)


def environment() -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in RECIPE_PACKAGES}
    spec = importlib.util.find_spec("atlasrank")
    if spec is None or spec.origin is None:
        raise ValueError("Pinned image atlasrank package is missing")
    encoder = Path(spec.origin).parent / "retrieval/encoder.py"
    return {
        "runtime_versions": versions,
        "runtime_platform": {"platform": platform.platform(), "machine": platform.machine()},
        "encoder_code_sha256": hashlib.sha256(encoder.read_bytes()).hexdigest(),
        "python": sys.version,
        "executable": sys.executable,
        "libc": platform.libc_ver(),
    }


def integer_file(paths: tuple[str, ...]) -> int | None:
    for name in paths:
        try:
            value = Path(name).read_text().strip()
            return int(value) if value != "max" else None
        except FileNotFoundError:
            continue
    return None


def resources() -> dict[str, Any]:
    import psutil

    limit = integer_file(
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    )
    used = integer_file(
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
    )
    quota = None
    try:
        value = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if value[0] != "max":
            quota = int(value[0]) / int(value[1])
    except FileNotFoundError:
        q = integer_file(("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",))
        period = integer_file(("/sys/fs/cgroup/cpu/cpu.cfs_period_us",))
        if q is not None and period and q > 0:
            quota = q / period
    affinity = len(psutil.Process().cpu_affinity())
    host = psutil.virtual_memory().available
    return {
        "host_available_bytes": host,
        "cgroup_memory_limit_bytes": limit,
        "cgroup_memory_current_bytes": used,
        "effective_available_bytes": effective_available(host, limit, used or 0),
        "cpu_affinity_count": affinity,
        "cpu_quota": quota,
        "effective_cpus": min(affinity, quota) if quota is not None else affinity,
    }


def inspect(
    original_recipe: Path | None, expected: str, allow_platform_difference: bool = False
) -> dict[str, Any]:
    require_compute(
        socket.gethostname(), sys.platform, bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    )
    report: dict[str, Any] = {
        "status": "blocked",
        "scope": "metadata_only_no_model_load",
        "environment": environment(),
        "resources": resources(),
        "expected_query_recipe_id": expected,
        "failures": [],
    }
    measured = report["resources"]
    if (
        measured["cgroup_memory_limit_bytes"] is None
        or measured["cgroup_memory_current_bytes"] is None
        or not 32 * 10**9 <= measured["cgroup_memory_limit_bytes"] <= 32 * 1024**3
    ):
        report["failures"].append("32 GB allocated RAM is not established")
    if measured["effective_cpus"] != 4:
        report["failures"].append("Exactly 4 allocated CPUs are not established")
    if measured["effective_available_bytes"] < 8 * 1024**3:
        report["failures"].append("Less than 8 GiB effective available RAM")
    if original_recipe is None or not original_recipe.is_file():
        report["failures"].append("Original M6 CPU query-recipe JSON is missing")
    else:
        original = json.loads(original_recipe.read_text())
        try:
            check_recipe(original, expected, report["environment"], allow_platform_difference)
            actual_recipe = {
                **original,
                **{
                    key: report["environment"][key]
                    for key in ("runtime_platform", "runtime_versions", "encoder_code_sha256")
                },
            }
            report["reference_recipe_evidence"] = "reconstructed_metadata_digest_verified"
            report["platform_difference_allowed"] = allow_platform_difference
            report["actual_query_recipe_id"] = digest(actual_recipe)
            report["actual_query_recipe"] = actual_recipe
            report["original_query_recipe_sha256"] = hashlib.sha256(
                original_recipe.read_bytes()
            ).hexdigest()
        except ValueError as error:
            report["failures"].append(str(error))
    if not report["failures"]:
        report["status"] = "passed_metadata_preflight_only"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-query-recipe", type=Path)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-platform-difference", action="store_true")
    args = parser.parse_args()
    # Refuse login before package queries, hashing or output creation.
    require_compute(
        socket.gethostname(), sys.platform, bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    )
    if args.output.exists():
        parser.error("Preserve existing evidence; choose a fresh output")
    spec = json.loads((args.descriptor / "spec.json").read_text())
    report = inspect(
        args.original_query_recipe, spec["query_recipe_id"], args.allow_platform_difference
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({"status": report["status"], "failures": report["failures"]}))
    return 0 if not report["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
