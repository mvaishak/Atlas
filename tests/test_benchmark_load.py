"""Client evidence must retain failures and distinguish their fast latency."""

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from atlasrank.benchmarking.load import run_trial
from atlasrank.benchmarking.measurements import distribution, summarize


def test_percentiles_include_fast_failures_separately() -> None:
    assert distribution([0.0, 10.0, 20.0, 30.0])["p95"] == pytest.approx(28.5)
    assert distribution([])["p99"] is None
    rows = [
        {
            "success": False,
            "status": 503,
            "error_code": "pipeline_busy",
            "timed_out": False,
            "timing_ms": {},
            "latency_ms": 1.0,
        }
        for _ in range(9)
    ] + [
        {
            "success": True,
            "status": 200,
            "error_code": None,
            "timed_out": False,
            "timing_ms": {"total": 90.0},
            "latency_ms": 100.0,
        }
    ]
    summary = summarize(rows, 2.0)
    assert summary["latency_ms"]["p95"] == pytest.approx(55.45)
    assert summary["success_latency_ms"]["p95"] == 100.0
    assert summary["error_latency_ms"]["p95"] == 1.0
    assert summary["throughput_success_rps"] == 0.5


@pytest.mark.parametrize("concurrency", [1, 4])
def test_http_errors_timeout_and_identity_are_preserved(tmp_path: Path, concurrency: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            query = body["query"]
            status = 200
            data: dict[str, Any] = {
                "pipeline_id": "bm25",
                "execution": "live",
                "bundle_id": "bundle",
                "versions": {"lexical": "lex-v1"},
                "timing_ms": {"total": 12.0},
                "returned": 1,
                "candidate_count": 2,
                "results": [{"secret": "product"}],
            }
            if query == "busy":
                status, data = 503, {"error": {"code": "pipeline_busy"}}
            elif query in ("server_limit", "unknown_limit"):
                self.send_response(503)
                self.end_headers()
                self.wfile.write(
                    b"Service Unavailable" if query == "server_limit" else b"Other overload"
                )
                return
            elif query == "timeout":
                time.sleep(0.15)
            elif query == "wrong":
                data["bundle_id"] = "unexpected"
            else:
                time.sleep(0.02)
            self.send_response(status)
            self.end_headers()
            try:
                self.wfile.write(json.dumps(data).encode())
            except BrokenPipeError:
                pass

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    queries = ["ok", "busy", "timeout", "wrong", "server_limit", "unknown_limit"]
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partition": "development",
                "queries": [
                    {
                        "query_id": str(i),
                        "query": q,
                        "query_sha256": hashlib.sha256(q.encode()).hexdigest(),
                    }
                    for i, q in enumerate(queries)
                ],
            }
        )
    )
    try:
        summary = run_trial(
            url=f"http://127.0.0.1:{server.server_port}",
            workload=workload,
            output=tmp_path / "out",
            pipeline="bm25",
            concurrency=concurrency,
            repeat=0,
            requests=12,
            warmup=1,
            timeout=0.05,
            deadline_utc=time.time() + 20,
            expected_bundle="bundle",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert summary["counts"] == {
        "attempts": 12,
        "succeeded": 2,
        "timeouts": 2,
        "transport": 0,
        "expected_busy": 2,
        "server_concurrency_limit": 2,
        "unexpected_errors": 4,
        "errors": 10,
    }
    assert summary["status_counts"] == {"200": 4, "503": 6, "none": 2}
    assert summary["stage_timing_ms"]["total"]["p95"] == 12.0
    assert summary["completed"] is True
    assert summary["success_latency_ms"]["p50"] > 15
    rows = sorted(
        [json.loads(line) for line in (tmp_path / "out/requests.jsonl").read_text().splitlines()],
        key=lambda row: row["request_index"],
    )
    assert [row["query_id"] for row in rows] == ["0", "1", "2", "3", "4", "5"] * 2
    assert rows[1]["error_code"] == "pipeline_busy"
    assert rows[2]["timed_out"] is True
    assert rows[3]["error_code"] == "response_identity"
    assert rows[4]["error_code"] == "server_concurrency_limit"
    assert rows[5]["error_code"] == "invalid_response"
    assert "product" not in (tmp_path / "out/requests.jsonl").read_text()


def test_deadline_preserves_empty_partial_trial(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partition": "development",
                "queries": [
                    {
                        "query_id": "q",
                        "query": "q",
                        "query_sha256": hashlib.sha256(b"q").hexdigest(),
                    }
                ],
            }
        )
    )
    summary = run_trial(
        url="http://127.0.0.1:1",
        workload=workload,
        output=tmp_path / "out",
        pipeline="bm25",
        concurrency=4,
        repeat=0,
        requests=10,
        warmup=2,
        timeout=0.01,
        deadline_utc=time.time() - 1,
    )
    assert summary["completed"] is False
    assert summary["unfinished_count"] == 10
    assert summary["counts"]["attempts"] == 0


def test_transport_errors_are_recorded_without_retries(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partition": "development",
                "queries": [
                    {
                        "query_id": "q",
                        "query": "q",
                        "query_sha256": hashlib.sha256(b"q").hexdigest(),
                    }
                ],
            }
        )
    )
    summary = run_trial(
        url=f"http://127.0.0.1:{port}/search",
        workload=workload,
        output=tmp_path / "out",
        pipeline="bm25",
        concurrency=4,
        repeat=0,
        requests=5,
        warmup=0,
        timeout=0.05,
        deadline_utc=time.time() + 10,
    )
    assert summary["counts"]["attempts"] == 5
    assert summary["counts"]["transport"] == 5
    assert summary["counts"]["succeeded"] == 0
    assert summary["status_counts"] == {"none": 5}


def test_integer_workload_ids_remain_in_request_evidence(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partition": "development",
                "queries": [
                    {"query_id": 42, "query": "q", "query_sha256": hashlib.sha256(b"q").hexdigest()}
                ],
            }
        )
    )
    summary = run_trial(
        url="http://127.0.0.1:1/search",
        workload=workload,
        output=tmp_path / "out",
        pipeline="bm25",
        concurrency=1,
        repeat=0,
        requests=1,
        warmup=0,
        timeout=0.05,
        deadline_utc=time.time() + 10,
    )
    assert summary["counts"]["attempts"] == 1
    assert json.loads((tmp_path / "out/requests.jsonl").read_text())["query_id"] == 42


def test_boolean_workload_id_is_rejected(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partition": "development",
                "queries": [
                    {
                        "query_id": True,
                        "query": "q",
                        "query_sha256": hashlib.sha256(b"q").hexdigest(),
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="query ID"):
        run_trial(
            url="http://127.0.0.1:1/search",
            workload=workload,
            output=tmp_path / "out",
            pipeline="bm25",
            concurrency=1,
            repeat=0,
            requests=1,
            warmup=0,
            timeout=0.05,
            deadline_utc=time.time() + 10,
        )
    assert not (tmp_path / "out").exists()
