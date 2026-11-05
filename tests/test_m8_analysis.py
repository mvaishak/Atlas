"""Independent returned-evidence checks and pooled latency calculations."""

import hashlib
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from atlasrank.benchmarking.measurements import summarize

analysis = runpy.run_path("scripts/analyze_m8_benchmark.py")
analyze_group = analysis["analyze_group"]
validate_rows = analysis["validate_rows"]
verify_checksums = analysis["verify_checksums"]


def test_checksums_detect_tampering_unlisted_files_and_unsafe_paths(tmp_path: Path) -> None:
    data = tmp_path / "result.json"
    data.write_text("{}")
    manifest = tmp_path / "checksums.json"
    checksum = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"result.json": checksum}))
    verify_checksums(tmp_path)
    data.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_checksums(tmp_path)
    data.write_text("{}")
    (tmp_path / "unlisted").write_text("extra")
    with pytest.raises(ValueError, match="coverage"):
        verify_checksums(tmp_path)
    manifest.write_text(json.dumps({"result.json": checksum, "../result.json": checksum}))
    with pytest.raises(ValueError, match="Unsafe"):
        verify_checksums(tmp_path)


def test_membership_rejects_wrong_query_and_duplicate_request_index() -> None:
    queries = [{"query_id": 42, "query_sha256": "a"}, {"query_id": 93, "query_sha256": "b"}]
    rows = [
        {
            "request_index": i,
            "query_id": query["query_id"],
            "query_sha256": query["query_sha256"],
            "pipeline": "bm25",
            "concurrency": 4,
            "repeat": 0,
            "success": False,
            "status": 503,
            "error_code": "pipeline_busy",
            "timed_out": False,
            "latency_ms": 1.0,
            "timing_ms": {},
        }
        for i, query in enumerate(queries)
    ]
    trial = {"pipeline": "bm25", "concurrency": 4, "repeat": 0}
    validate_rows(rows, queries, trial)
    rows[1]["query_id"] = 42
    with pytest.raises(ValueError, match="query membership"):
        validate_rows(rows, queries, trial)
    rows[1]["query_id"] = 93
    rows[1]["request_index"] = 0
    with pytest.raises(ValueError, match="request indexes"):
        validate_rows(rows, queries, trial)


@pytest.mark.parametrize("mutation", ["missing", "negative", "nonfinite", "bm25_encoding"])
def test_successful_rows_require_valid_pipeline_stage_inventory(mutation: str) -> None:
    timings = {
        "lexical": 1.0,
        "total": 1.0,
        "request_dispatch": 1.0,
        "query_encoding_total": 0.0,
        "feature_query_encoding": 0.0,
    }
    if mutation == "missing":
        del timings["lexical"]
    elif mutation == "negative":
        timings["lexical"] = -1.0
    elif mutation == "nonfinite":
        timings["lexical"] = float("inf")
    else:
        timings["query_encoding_total"] = 1.0
    trial = {
        "pipeline": "bm25",
        "concurrency": 1,
        "repeat": 0,
        "bundle_id": "bundle",
        "versions": {"lexical": "v1"},
    }
    row = {
        **trial,
        "request_index": 0,
        "query_id": 42,
        "query_sha256": "hash",
        "success": True,
        "status": 200,
        "error_code": None,
        "timed_out": False,
        "latency_ms": 1.0,
        "timing_ms": timings,
    }
    with pytest.raises(ValueError, match="stage"):
        validate_rows([row], [{"query_id": 42, "query_sha256": "hash"}], trial)


def test_pooled_success_percentiles_and_failure_rates_do_not_average_percentiles() -> None:
    rows: list[dict[str, Any]] = [
        {
            "success": False,
            "status": 503,
            "error_code": "pipeline_busy",
            "timed_out": False,
            "latency_ms": 1.0,
            "timing_ms": {},
        }
        for _ in range(9)
    ]
    rows += [
        {
            "success": True,
            "status": 200,
            "error_code": None,
            "timed_out": False,
            "latency_ms": 100.0,
            "timing_ms": {"total": 90.0},
        }
    ]
    summary = {"repeat": 0, "measured_wall_seconds": 2.0, "throughput_success_rps": 0.5}
    result = analyze_group([(summary, rows)])
    assert result["latency_ms"]["p95"] == pytest.approx(55.45)
    assert result["success_latency_ms"]["p95"] == 100.0
    assert result["rates"]["errors"] == 0.9
    assert result["rates"]["expected_busy"] == 0.9
    assert result["counts"]["attempts"] == 10
    assert result["stage_timing_ms"]["total"]["p95"] == 90.0
    assert result["repeats"][0]["throughput_success_rps"] == 0.5
    assert "throughput_success_rps" not in result


@pytest.fixture
def synthetic_run(tmp_path: Path) -> tuple[Path, Path, Callable[[], None]]:
    root = tmp_path / "run"
    root.mkdir()
    query_hash = hashlib.sha256(b"q").hexdigest()
    queries = [{"query_id": i, "query": "q", "query_sha256": query_hash} for i in range(1000)]
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps({"schema_version": 1, "partition": "development", "queries": queries})
    )
    workload_hash = hashlib.sha256(workload.read_bytes()).hexdigest()
    trials = []
    for pipeline in ("bm25", "rrf_m4", "m6_xgb_m4"):
        for concurrency in (1, 4, 8):
            for repeat in (0, 1, 2):
                trial = {
                    "pipeline": pipeline,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "bundle_id": "bundle",
                    "versions": {"lexical": "v1"},
                }
                timings = {
                    "lexical": 1.0,
                    "total": 1.0,
                    "request_dispatch": 1.0,
                    "query_encoding_total": 0.0 if pipeline == "bm25" else 1.0,
                    "feature_query_encoding": 1.0 if pipeline == "m6_xgb_m4" else 0.0,
                }
                if pipeline != "bm25":
                    timings.update(encoding=1.0, dense_retrieval=1.0, fusion=1.0)
                if pipeline == "m6_xgb_m4":
                    timings.update(
                        feature_construction=1.0,
                        ranker_prediction=1.0,
                        ranking_order=1.0,
                        features_and_prediction=1.0,
                    )
                rows = [
                    {
                        **trial,
                        "request_index": i,
                        "query_id": i,
                        "query_sha256": query_hash,
                        "success": True,
                        "status": 200,
                        "error_code": None,
                        "timed_out": False,
                        "latency_ms": float(i + 1),
                        "timing_ms": timings,
                    }
                    for i in range(1000)
                ]
                summary = {
                    **trial,
                    **summarize(rows, 2.0),
                    "completed": True,
                    "requested_count": 1000,
                    "warmup_count": 100,
                    "warmup_errors": 0,
                    "workload_sha256": workload_hash,
                }
                directory = root / f"{pipeline}-c{concurrency}-r{repeat}"
                directory.mkdir()
                (directory / "requests.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                (directory / "warmup.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows[:100])
                )
                (directory / "summary.json").write_text(json.dumps(summary))
                trials.append(summary)
    (root / "result.json").write_text(
        json.dumps(
            {
                "status": "completed_reference_benchmark",
                "reason": None,
                "trials": trials,
                "server_exit_code": 0,
                "shutdown": {
                    "shutdown_ok": True,
                    "model_sha256_after": "model",
                    "reported_peak_process_rss_bytes": 100,
                },
                "sampled_peak_tree_rss_bytes": 200,
                "new_storage_bytes": 300,
                "wall_seconds": 20.0,
            }
        )
    )
    (root / "environment.json").write_text(
        json.dumps({"source_manifest_sha256": "source", "workload_sha256": workload_hash})
    )
    (root / "server-ready.json").write_text(
        json.dumps(
            {
                "instrumented_real_http_parity": "passed_exact_features_and_predictions",
                "readiness_response": {"ready": True, "bundle_id": "bundle"},
                "versions": {"lexical": "v1"},
                "model_sha256_before": "model",
            }
        )
    )
    (root / "server-shutdown.json").write_text(
        json.dumps(json.loads((root / "result.json").read_text())["shutdown"])
    )

    def seal() -> None:
        checksums = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file() and path.name != "checksums.json"
        }
        (root / "checksums.json").write_text(json.dumps(checksums))

    seal()
    return root, workload, seal


def test_complete_inventory_is_verified_before_reporting(
    synthetic_run: tuple[Path, Path, Callable[[], None]],
) -> None:
    root, workload, seal = synthetic_run
    report = analysis["analyze"](root, workload)
    assert report["trial_count"] == 27
    assert len(report["groups"]) == 9
    assert {(group["pipeline"], group["concurrency"]) for group in report["groups"]} == {
        (pipeline, concurrency)
        for pipeline in ("bm25", "rrf_m4", "m6_xgb_m4")
        for concurrency in (1, 4, 8)
    }
    assert report["groups"][0]["counts"]["attempts"] == 3000
    assert report["groups"][0]["success_latency_ms"]["p95"] == pytest.approx(950.05)
    rows_path = root / "bm25-c1-r0/requests.jsonl"
    rows_path.write_text("\n".join(rows_path.read_text().splitlines()[1:]) + "\n")
    seal()
    with pytest.raises(ValueError, match="request indexes"):
        analysis["analyze"](root, workload)


@pytest.mark.parametrize(
    "filename,keys,value",
    [
        ("result.json", ("server_exit_code",), 1),
        ("result.json", ("shutdown", "shutdown_ok"), False),
        ("result.json", ("shutdown", "model_sha256_after"), "changed"),
        ("result.json", ("shutdown", "reported_peak_process_rss_bytes"), 12 * 1024**3 + 1),
        ("result.json", ("sampled_peak_tree_rss_bytes",), 12 * 1024**3 + 1),
        ("result.json", ("new_storage_bytes",), 1024**3 + 1),
        ("result.json", ("wall_seconds",), 14401),
        ("server-ready.json", ("readiness_response", "ready"), False),
        ("server-ready.json", ("instrumented_real_http_parity",), "unverified"),
        ("server-ready.json", ("readiness_response", "bundle_id"), "different"),
        ("server-ready.json", ("versions", "lexical"), "different"),
        ("environment.json", ("workload_sha256",), "different"),
    ],
)
def test_resealed_run_cannot_waive_resource_or_readiness_failures(
    synthetic_run: tuple[Path, Path, Callable[[], None]],
    filename: str,
    keys: tuple[str, ...],
    value: Any,
) -> None:
    root, workload, seal = synthetic_run
    path = root / filename
    payload = json.loads(path.read_text())
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(payload))
    if filename == "result.json":
        (root / "server-shutdown.json").write_text(json.dumps(payload["shutdown"]))
    seal()
    with pytest.raises(ValueError):
        analysis["analyze"](root, workload)


@pytest.mark.parametrize("tamper", ["missing", "wrong_query", "unreported_error"])
def test_warmup_is_verified_from_raw_rows(
    synthetic_run: tuple[Path, Path, Callable[[], None]],
    tamper: str,
) -> None:
    root, workload, seal = synthetic_run
    path = root / "bm25-c1-r0/warmup.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if tamper == "missing":
        rows.pop()
    elif tamper == "wrong_query":
        rows[0]["query_id"] = 999
    else:
        rows[0].update(success=False, status=503, error_code="pipeline_busy")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    seal()
    with pytest.raises(ValueError):
        analysis["analyze"](root, workload)


@pytest.mark.parametrize(
    "pipeline,stage,missing",
    [
        ("rrf_m4", "fusion", True),
        ("m6_xgb_m4", "ranker_prediction", True),
        ("rrf_m4", "feature_query_encoding", False),
    ],
)
def test_neural_success_stages_and_zero_encoding_contracts(
    synthetic_run: tuple[Path, Path, Callable[[], None]],
    pipeline: str,
    stage: str,
    missing: bool,
) -> None:
    root, workload, seal = synthetic_run
    path = root / f"{pipeline}-c1-r0/requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if missing:
        del rows[0]["timing_ms"][stage]
    else:
        rows[0]["timing_ms"][stage] = 1.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    seal()
    with pytest.raises(ValueError, match="stage"):
        analysis["analyze"](root, workload)


def test_zero_duration_clock_readings_are_valid(
    synthetic_run: tuple[Path, Path, Callable[[], None]],
) -> None:
    root, _, _ = synthetic_run
    directory = root / "m6_xgb_m4-c1-r0"
    trial = json.loads((directory / "summary.json").read_text())
    row = json.loads((directory / "requests.jsonl").read_text().splitlines()[0])
    row["timing_ms"] = dict.fromkeys(row["timing_ms"], 0.0)
    validate_rows(
        [row], [{"query_id": row["query_id"], "query_sha256": row["query_sha256"]}], trial
    )
