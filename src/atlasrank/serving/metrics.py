"""Bounded aggregate exposition and request body admission; no payload logging."""

import json
import logging
import threading
import uuid
from collections import defaultdict

from starlette.types import ASGIApp, Message, Receive, Scope, Send

ROUTES = frozenset(
    {
        "/search",
        "/rerank",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/demo",
        "/demo/tokens.css",
        "/comparison",
    }
)
PIPELINES = frozenset({"bm25", "dense_m4", "rrf_m4", "m6_xgb_m4"})


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.duration: dict[str, float] = defaultdict(float)
        self.stage_sum: dict[tuple[str, str], float] = defaultdict(float)
        self.stage_count: dict[tuple[str, str], int] = defaultdict(int)

    def observe(self, route: str, status: int, seconds: float) -> None:
        route = route if route in ROUTES else "other"
        with self.lock:
            self.requests[(route, status)] += 1
            self.duration[route] += seconds

    def stages(self, pipeline: str, values: dict[str, float]) -> None:
        if pipeline not in PIPELINES:
            return
        allowed = {
            "lexical",
            "encoding",
            "dense_retrieval",
            "fusion",
            "features_and_prediction",
            "feature_construction",
            "ranker_prediction",
            "ranking_order",
            "request_dispatch",
            "total",
        }
        with self.lock:
            for name, value in values.items():
                if name in allowed:
                    key = (pipeline, name)
                    self.stage_sum[key] += value / 1000
                    self.stage_count[key] += 1

    def render(self) -> str:
        lines = [
            "# TYPE atlasrank_requests_total counter",
            "# TYPE atlasrank_request_duration_seconds_sum counter",
            "# TYPE atlasrank_stage_seconds_sum counter",
            "# TYPE atlasrank_stage_seconds_count counter",
        ]
        with self.lock:
            for (route, status), count in sorted(self.requests.items()):
                lines.append(
                    f'atlasrank_requests_total{{route="{route}",status="{status}"}} {count}'
                )
            for route, value in sorted(self.duration.items()):
                lines.append(f'atlasrank_request_duration_seconds_sum{{route="{route}"}} {value}')
            for (pipeline, stage), value in sorted(self.stage_sum.items()):
                labels = f'pipeline="{pipeline}",stage="{stage}"'
                lines.append(f"atlasrank_stage_seconds_sum{{{labels}}} {value}")
                lines.append(
                    f"atlasrank_stage_seconds_count{{{labels}}} "
                    f"{self.stage_count[(pipeline, stage)]}"
                )
        return "\n".join(lines) + "\n"


class RequestBodyLimit:
    """Buffer at most 16 KiB before JSON parsing, including chunked bodies."""

    def __init__(self, app: ASGIApp, metrics: Metrics, maximum: int = 16384) -> None:
        self.app, self.metrics, self.maximum = app, metrics, maximum

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.maximum:
                request_id = str(scope.get("state", {}).get("request_id") or uuid.uuid4().hex)
                logging.getLogger("atlasrank.http").info(
                    json.dumps(
                        {
                            "event": "request_rejected",
                            "request_id": request_id,
                            "status": 413,
                            "error_type": "body_too_large",
                        }
                    )
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"x-request-id", request_id.encode()),
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error":{"code":"request_too_large"}}',
                    }
                )
                return
            if not message.get("more_body", False):
                break
        consumed = False

        async def replay() -> Message:
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)
