"""Explicit linear-interpolation latency distributions without NumPy."""

from collections import Counter
from typing import Any


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    result: dict[str, float | int | None] = {"count": len(values)}
    for name, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        if not ordered:
            result[name] = None
            continue
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


def summarize(rows: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    successful = [row for row in rows if row["success"]]
    errors = [row for row in rows if not row["success"]]
    stages = sorted({name for row in successful for name in row["timing_ms"]})
    busy = sum(row["status"] == 503 and row["error_code"] == "pipeline_busy" for row in rows)
    server_limit = sum(
        row["status"] == 503 and row["error_code"] == "server_concurrency_limit" for row in rows
    )
    timeouts = sum(row["timed_out"] for row in rows)
    transport = sum(row["error_code"] == "transport" for row in rows)
    return {
        "counts": {
            "attempts": len(rows),
            "succeeded": len(successful),
            "errors": len(errors),
            "timeouts": timeouts,
            "transport": transport,
            "expected_busy": busy,
            "server_concurrency_limit": server_limit,
            "unexpected_errors": len(errors) - busy - server_limit - timeouts - transport,
        },
        "status_counts": dict(
            Counter("none" if r["status"] is None else str(r["status"]) for r in rows)
        ),
        "latency_ms": distribution([row["latency_ms"] for row in rows]),
        "success_latency_ms": distribution([row["latency_ms"] for row in successful]),
        "error_latency_ms": distribution([row["latency_ms"] for row in errors]),
        "stage_timing_ms": {
            stage: distribution(
                [row["timing_ms"][stage] for row in successful if stage in row["timing_ms"]]
            )
            for stage in stages
        },
        "measured_wall_seconds": wall_seconds,
        "throughput_success_rps": len(successful) / wall_seconds if wall_seconds > 0 else 0.0,
    }
