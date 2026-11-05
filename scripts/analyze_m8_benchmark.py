"""Verify returned M8 evidence and pool request observations, never percentiles."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from atlasrank.benchmarking.measurements import distribution, summarize


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checksums(root: Path) -> None:
    manifest = read(root / "checksums.json")
    if not isinstance(manifest, dict) or "result.json" not in manifest:
        raise ValueError("Missing checksum manifest result")
    for name, expected in manifest.items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or name == "checksums.json":
            raise ValueError("Unsafe checksum path")
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Unsafe checksum symlink")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"File checksum differs: {name}")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Unsafe evidence symlink")
        if path.is_file() and path != root / "checksums.json":
            actual.add(str(path.relative_to(root)))
    if set(manifest) != actual:
        raise ValueError("Checksum coverage differs from evidence files")


def equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            equivalent(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            equivalent(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if type(actual) in (float, int) and type(expected) in (float, int):
        return (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    return type(actual) is type(expected) and actual == expected


def validate_rows(
    rows: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    trial: dict[str, Any],
    expected_count: int | None = None,
) -> None:
    expected_count = len(queries) if expected_count is None else expected_count
    required = {
        "lexical",
        "total",
        "request_dispatch",
        "query_encoding_total",
        "feature_query_encoding",
    }
    if trial["pipeline"] != "bm25":
        required.update(("encoding", "dense_retrieval", "fusion"))
    if trial["pipeline"] == "m6_xgb_m4":
        required.update(
            (
                "feature_construction",
                "ranker_prediction",
                "ranking_order",
                "features_and_prediction",
            )
        )
    indexes = [row["request_index"] for row in rows]
    if any(type(i) is not int for i in indexes) or sorted(indexes) != list(range(expected_count)):
        raise ValueError("Invalid request indexes")
    for row in rows:
        query = queries[row["request_index"] % len(queries)]
        if any(row[key] != query[key] for key in ("query_id", "query_sha256")):
            raise ValueError("Incorrect query membership")
        if any(row[key] != trial[key] for key in ("pipeline", "concurrency", "repeat")):
            raise ValueError("Incorrect trial membership")
        if type(row["success"]) is not bool or type(row["timed_out"]) is not bool:
            raise ValueError("Invalid request outcome")
        if not math.isfinite(row["latency_ms"]) or row["latency_ms"] < 0:
            raise ValueError("Invalid latency")
        if row["success"] and (
            row["status"] != 200 or row["error_code"] is not None or row["timed_out"]
        ):
            raise ValueError("Inconsistent successful outcome")
        if row["success"] and any(row[key] != trial[key] for key in ("bundle_id", "versions")):
            raise ValueError("Response identity differs")
        if row["success"]:
            timings = row["timing_ms"]
            if (
                not isinstance(timings, dict)
                or not required.issubset(timings)
                or any(
                    type(value) not in (float, int) or not math.isfinite(value) or value < 0
                    for value in timings.values()
                )
            ):
                raise ValueError("Invalid successful stage inventory or timing")
            zero = (
                ("query_encoding_total", "feature_query_encoding")
                if trial["pipeline"] == "bm25"
                else (("feature_query_encoding",) if trial["pipeline"] == "rrf_m4" else ())
            )
            if any(timings[stage] != 0 for stage in zero):
                raise ValueError("Unexpected encoding stage time")


def analyze_group(trials: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> dict[str, Any]:
    rows = [row for _, observations in trials for row in observations]
    pooled = summarize(rows, 1.0)
    del pooled["throughput_success_rps"], pooled["measured_wall_seconds"]
    counts = pooled["counts"]
    pooled["rates"] = {
        key: value / counts["attempts"] for key, value in counts.items() if key != "attempts"
    }
    pooled["repeats"] = [
        summary for summary, _ in sorted(trials, key=lambda item: item[0]["repeat"])
    ]
    return pooled


def analyze(root: Path, workload_path: Path) -> dict[str, Any]:
    verify_checksums(root)
    result, workload = read(root / "result.json"), read(workload_path)
    if result["status"] != "completed_reference_benchmark" or result["reason"] is not None:
        raise ValueError("Run did not complete")
    queries = workload["queries"]
    if (
        workload["schema_version"] != 1
        or workload["partition"] != "development"
        or len(queries) != 1000
    ):
        raise ValueError("Expected 1000 development queries")
    for query in queries:
        if hashlib.sha256(query["query"].encode()).hexdigest() != query["query_sha256"]:
            raise ValueError("Workload query hash differs")
    inventory = set(itertools.product(("bm25", "rrf_m4", "m6_xgb_m4"), (1, 4, 8), (0, 1, 2)))
    keys = [
        (trial["pipeline"], trial["concurrency"], trial["repeat"]) for trial in result["trials"]
    ]
    if len(keys) != 27 or set(keys) != inventory:
        raise ValueError("Incomplete or duplicate 27-trial inventory")
    groups: dict[tuple[str, int], list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
    workload_hash = digest(workload_path)
    environment, ready = read(root / "environment.json"), read(root / "server-ready.json")
    shutdown = result["shutdown"]
    if (
        result["server_exit_code"] != 0
        or shutdown["shutdown_ok"] is not True
        or not equivalent(shutdown, read(root / "server-shutdown.json"))
        or ready["readiness_response"]["ready"] is not True
        or ready["instrumented_real_http_parity"] != "passed_exact_features_and_predictions"
        or not ready["model_sha256_before"]
        or ready["model_sha256_before"] != shutdown["model_sha256_after"]
        or environment["workload_sha256"] != workload_hash
    ):
        raise ValueError("Run readiness, shutdown, model, or workload evidence differs")
    for value, cap in (
        (shutdown["reported_peak_process_rss_bytes"], 12 * 1024**3),
        (result["sampled_peak_tree_rss_bytes"], 12 * 1024**3),
        (result["new_storage_bytes"], 1024**3),
        (result["wall_seconds"], 14400),
    ):
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= cap:
            raise ValueError("Run resource evidence exceeds approved limits")
    for recorded, (pipeline, concurrency, repeat) in zip(result["trials"], keys, strict=True):
        directory = root / f"{pipeline}-c{concurrency}-r{repeat}"
        summary = read(directory / "summary.json")
        if (
            summary["bundle_id"] != ready["readiness_response"]["bundle_id"]
            or summary["versions"] != ready["versions"]
        ):
            raise ValueError("Trial identity differs from ready server")
        if not equivalent(recorded, summary):
            raise ValueError("Run result and trial summary differ")
        if (
            summary["completed"] is not True
            or summary["requested_count"] != 1000
            or summary["warmup_count"] != 100
            or summary["counts"]["attempts"] != 1000
            or summary["workload_sha256"] != workload_hash
        ):
            raise ValueError("Trial contract differs")
        rows = [
            json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        validate_rows(rows, queries, summary)
        warmup = [
            json.loads(line) for line in (directory / "warmup.jsonl").read_text().splitlines()
        ]
        validate_rows(warmup, queries, summary, expected_count=100)
        if summary["warmup_errors"] != sum(not row["success"] for row in warmup):
            raise ValueError("Warmup error count differs")
        seconds = summary["measured_wall_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid measured wall time")
        for key, value in summarize(rows, seconds).items():
            if not equivalent(summary[key], value):
                raise ValueError(f"Recomputed summary differs: {key}")
        groups.setdefault((pipeline, concurrency), []).append((summary, rows))
    return {
        "schema_version": 1,
        "status": "verified_reference_benchmark",
        "trial_count": 27,
        "result_sha256": digest(root / "result.json"),
        "checksums_sha256": digest(root / "checksums.json"),
        "workload_sha256": workload_hash,
        "source_manifest_sha256": environment["source_manifest_sha256"],
        "readiness": ready,
        "environment": environment,
        "resources": {key: value for key, value in result.items() if key != "trials"},
        "workload": {
            "query_count": len(queries),
            "character_length": distribution([float(len(q["query"])) for q in queries]),
            "whitespace_token_length": distribution(
                [float(len(q["query"].split())) for q in queries]
            ),
            "metadata": {key: value for key, value in workload.items() if key != "queries"},
        },
        "groups": [
            {"pipeline": pipeline, "concurrency": concurrency, **analyze_group(trials)}
            for (pipeline, concurrency), trials in sorted(groups.items())
        ],
        "limitations": [
            "Shared cluster node and uncontrolled load; "
            "CPU quota is not dedicated core reservation.",
            "Pooled percentiles use raw observations; "
            "throughput is reported separately per repeat.",
            "Server stages overlap; their sum does not establish exclusive CPU time.",
            "Client minus server timing includes networking, serialization and middleware.",
            "Request dispatch includes validation and thread-pool scheduling, not pure queue time.",
            "Development performance evidence only; "
            "no quality evaluation or optimization comparison.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.workload)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
