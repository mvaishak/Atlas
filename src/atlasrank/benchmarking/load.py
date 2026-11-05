"""Separate-process, closed-loop HTTP load generator with bounded evidence."""

import argparse
import hashlib
import json
import math
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TextIO

from atlasrank.benchmarking.measurements import summarize

PIPELINES = ("bm25", "rrf_m4", "m6_xgb_m4")


def _queries(path: Path) -> list[dict[str, Any]]:
    workload = json.loads(path.read_text())
    if workload.get("schema_version") != 1 or workload.get("partition") != "development":
        raise ValueError("Expected schema 1 development workload")
    queries = workload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("Workload needs queries")
    for query in queries:
        if not isinstance(query, dict) or any(
            not isinstance(query.get(key), str) or not query[key]
            for key in ("query", "query_sha256")
        ):
            raise ValueError("Invalid workload query")
        query_id = query.get("query_id")
        if isinstance(query_id, bool) or not isinstance(query_id, (str, int)) or query_id == "":
            raise ValueError("Invalid workload query ID")
        if hashlib.sha256(query["query"].encode()).hexdigest() != query["query_sha256"]:
            raise ValueError("Query hash mismatch")
    return queries


def _request(url: str, query: dict[str, Any], pipeline: str, timeout: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "query_id": query["query_id"],
        "query_sha256": query["query_sha256"],
        "success": False,
        "status": None,
        "error_code": None,
        "timed_out": False,
        "timing_ms": {},
    }
    payload = json.dumps({"query": query["query"], "pipeline": pipeline, "top_k": 10}).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    body: dict[str, Any] = {}
    try:
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            row["status"] = response.code
            row["request_id"] = response.headers.get("X-Request-ID")
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("Response too large")
            parsed = (
                {"error": {"code": "server_concurrency_limit"}}
                if row["status"] == 503 and raw == b"Service Unavailable"
                else json.loads(raw)
            )
            if not isinstance(parsed, dict):
                raise ValueError("Expected response object")
            body = parsed
        if row["status"] == 200:
            if (
                body.get("execution") != "live"
                or body.get("pipeline_id") != pipeline
                or not isinstance(body.get("bundle_id"), str)
                or not body["bundle_id"]
                or not isinstance(body.get("versions"), dict)
            ):
                raise ValueError("Invalid response identity")
            timings = body.get("timing_ms")
            if not isinstance(timings, dict) or any(
                not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
                for v in timings.values()
            ):
                raise ValueError("Invalid response timing")
            for key in ("bundle_id", "versions", "candidate_count", "returned", "timing_ms"):
                row[key] = body.get(key)
            row["success"] = True
        else:
            error_body = body.get("error", {})
            row["error_code"] = error_body.get("code", "http_error")
    except TimeoutError:
        row.update(error_code="timeout", timed_out=True)
    except urllib.error.URLError as error:
        timed_out = isinstance(error.reason, (TimeoutError, socket.timeout))
        row.update(error_code="timeout" if timed_out else "transport", timed_out=timed_out)
    except (ValueError, TypeError, AttributeError):
        row["error_code"] = "invalid_response"
    except OSError:
        row["error_code"] = "transport"
    row["latency_ms"] = (time.perf_counter() - started) * 1000
    return row


def run_trial(
    *,
    url: str,
    workload: Path,
    output: Path,
    pipeline: str,
    concurrency: int,
    repeat: int,
    requests: int,
    warmup: int,
    timeout: float,
    deadline_utc: float,
    expected_bundle: str | None = None,
) -> dict[str, Any]:
    """Write every completed attempt immediately, including errors and warmup."""
    if (
        pipeline not in PIPELINES
        or concurrency not in (1, 4, 8)
        or repeat not in (0, 1, 2)
        or requests <= 0
        or warmup < 0
        or timeout <= 0
        or not math.isfinite(timeout)
        or not math.isfinite(deadline_utc)
    ):
        raise ValueError("Invalid trial settings")
    queries = _queries(workload)
    output.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    identity: dict[str, Any] = {}
    warmup_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def attempt(index: int, stream: TextIO, measured: bool) -> bool:
        remaining = deadline_utc - time.time()
        if remaining <= 0:
            return False
        row = _request(url, queries[index % len(queries)], pipeline, min(timeout, remaining))
        row.update(request_index=index, pipeline=pipeline, concurrency=concurrency, repeat=repeat)
        with lock:
            if row["success"]:
                observed = {"bundle_id": row["bundle_id"], "versions": row["versions"]}
                if (expected_bundle is not None and row["bundle_id"] != expected_bundle) or (
                    identity and identity != observed
                ):
                    row.update(success=False, error_code="response_identity")
                else:
                    identity.update(observed)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            (rows if measured else warmup_rows).append(row)
        return True

    with (output / "warmup.jsonl").open("x") as stream:
        for index in range(warmup):
            if not attempt(index, stream, False):
                break
    started = time.perf_counter()
    with (output / "requests.jsonl").open("x") as stream:
        if len(warmup_rows) == warmup:

            def worker(worker_index: int) -> None:
                for index in range(worker_index, requests, concurrency):
                    if not attempt(index, stream, True):
                        break

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(worker, range(concurrency)))
    summary = summarize(rows, time.perf_counter() - started)
    completed = len(rows) == requests and len(warmup_rows) == warmup
    summary.update(
        schema_version=1,
        pipeline=pipeline,
        concurrency=concurrency,
        repeat=repeat,
        requested_count=requests,
        warmup_count=len(warmup_rows),
        warmup_errors=sum(not row["success"] for row in warmup_rows),
        completed=completed,
        status="complete" if completed else "interrupted",
        unfinished_count=requests - len(rows),
        deadline_utc=deadline_utc,
        workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        **identity,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full /search HTTP endpoint")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", choices=PIPELINES, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--deadline-utc", type=float, required=True)
    parser.add_argument("--expected-bundle")
    args = parser.parse_args()
    summary = run_trial(**vars(args))
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
