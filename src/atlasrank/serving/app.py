"""One immutable lexical bundle behind a bounded local search contract."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlasrank.artifacts.lexical import InvalidIndex
from atlasrank.artifacts.serving_bundle import Pipeline
from atlasrank.data.common import peak_rss_bytes
from atlasrank.retrieval.analyzer import analyze
from atlasrank.retrieval.bm25 import LexicalIndex, load
from atlasrank.serving.comparison import Comparison, load_comparison
from atlasrank.serving.metrics import ROUTES, Metrics, RequestBodyLimit
from atlasrank.serving.runtime import (
    PipelineBusy,
    PipelineRuntime,
    PipelineUnavailable,
    load_runtime,
)

if TYPE_CHECKING:
    from atlasrank.serving.ranking import RankingRuntime

LOGGER = logging.getLogger("atlasrank.http")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=512)
    top_k: int = Field(default=10, ge=1, le=100)
    locale: Literal["us"] = "us"
    pipeline: Pipeline = "bm25"

    @field_validator("query")
    @classmethod
    def meaningful(cls, query: str) -> str:
        tokens = analyze(query)
        if not tokens or len(tokens) > 128:
            raise ValueError("Query must contain 1 to 128 letter/digit tokens")
        return query


@dataclass
class State:
    index: LexicalIndex | None = None
    comparison: Comparison | None = None
    ranking: RankingRuntime | None = None
    serving: PipelineRuntime | None = None


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query_id: int = Field(ge=0)


class RerankRequest(SearchRequest):
    product_ids: list[str] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_ids(self) -> RerankRequest:
        if len(set(self.product_ids)) != len(self.product_ids) or any(
            not x.strip() for x in self.product_ids
        ):
            raise ValueError("Unique nonempty candidate IDs required")
        return self


def create_app(
    index_path: Path | None = None,
    comparison_path: Path | None = None,
    comparison_sha256: str | None = None,
    *,
    ranking: RankingRuntime | None = None,
    serving_runtime: PipelineRuntime | None = None,
    serving_bundle: Path | None = None,
    artifact_root: Path | None = None,
) -> FastAPI:
    state = State()
    metrics = Metrics()
    complete_configured = serving_runtime is not None or serving_bundle is not None
    if serving_runtime is not None and serving_bundle is not None:
        raise ValueError("Choose runtime or bundle, not both")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if complete_configured:
            try:
                runtime = serving_runtime
                if runtime is None:
                    if serving_bundle is None or artifact_root is None:
                        raise ValueError("Serving descriptor requires artifact root")
                    runtime = load_runtime(serving_bundle, artifact_root)
                runtime.validate()
                state.serving, state.index = runtime, runtime.lexical
                state.ranking = runtime.ranking
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                MemoryError,
                ImportError,
            ):
                LOGGER.error(
                    json.dumps(
                        {"event": "serving_unavailable", "error_type": "artifact_validation"}
                    )
                )
        try:
            if index_path is not None and not complete_configured:
                state.index = load(index_path)
                if peak_rss_bytes() > 6 * 1024**3:
                    state.index = None
                    raise InvalidIndex("Serving load exceeds the 6 GiB process ceiling")
        except InvalidIndex:
            state.index = None
            LOGGER.error(
                json.dumps({"event": "index_unavailable", "error_type": "artifact_validation"})
            )
        if comparison_path is not None and state.index is not None:
            try:
                state.comparison = load_comparison(
                    comparison_path, comparison_sha256, state.index.bundle_id
                )
            except (OSError, ValueError, KeyError, TypeError):
                LOGGER.error(
                    json.dumps(
                        {"event": "comparison_unavailable", "error_type": "artifact_validation"}
                    )
                )
        if ranking is not None and state.index is not None:
            try:
                ranking.validate(state.index)
                state.ranking = ranking
            except ValueError:
                LOGGER.error(
                    json.dumps({"event": "ranker_unavailable", "error_type": "artifact_validation"})
                )
        yield
        state.serving = None
        state.ranking = None
        state.index = None
        state.comparison = None

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    app.add_middleware(RequestBodyLimit, metrics=metrics)

    @app.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.error(
                json.dumps(
                    {"event": "request_failed", "request_id": request_id, "error_type": "internal"}
                )
            )
            response = JSONResponse(status_code=500, content={"error": {"code": "internal_error"}})
        route = request.url.path if request.url.path in ROUTES else "other"
        metrics.observe(route, response.status_code, time.perf_counter() - started)
        LOGGER.info(
            json.dumps(
                {
                    "event": "request_completed",
                    "request_id": request_id,
                    "route": route,
                    "status": response.status_code,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "bundle_id": state.index.bundle_id if state.index else None,
                }
            )
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request"}})

    @app.get("/metrics")
    def aggregate_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/health/live")
    def live() -> dict[str, bool]:
        return {"live": True}

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        if state.index is None or (complete_configured and state.serving is None):
            return JSONResponse(
                status_code=503, content={"ready": False, "error": {"code": "index_unavailable"}}
            )
        return JSONResponse(
            {
                "ready": True,
                "bundle_id": state.serving.bundle_id if state.serving else state.index.bundle_id,
            }
        )

    @app.get("/demo/tokens.css")
    def demo_tokens() -> Response:
        css = files("atlasrank.serving").joinpath("static/tokens.css").read_text()
        return Response(css, media_type="text/css", headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/demo")
    def demo() -> HTMLResponse:
        html = files("atlasrank.serving").joinpath("static/demo.html").read_text()
        hashes = {}
        for tag in ("script", "style"):
            content = re.search(f"<{tag}>(.*?)</{tag}>", html, re.DOTALL)
            assert content is not None
            hashes[tag] = base64.b64encode(hashlib.sha256(content[1].encode()).digest()).decode()
        policy = (
            "default-src 'none'; connect-src 'self'; "
            f"script-src 'sha256-{hashes['script']}'; style-src 'self' 'sha256-{hashes['style']}'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        return HTMLResponse(
            html,
            headers={
                "Content-Security-Policy": policy,
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )

    @app.get("/comparison")
    def examples() -> JSONResponse:
        if state.comparison is None:
            return JSONResponse(
                status_code=503, content={"error": {"code": "comparison_unavailable"}}
            )
        return JSONResponse(
            {
                "mode": "saved_development",
                "fixture_id": state.comparison.fixture_id,
                "examples": [
                    {"query_id": e.query_id, "query": e.query} for e in state.comparison.examples
                ],
            }
        )

    @app.post("/comparison")
    def comparison(payload: ComparisonRequest) -> JSONResponse:
        started = time.perf_counter()
        if state.comparison is None:
            return JSONResponse(
                status_code=503, content={"error": {"code": "comparison_unavailable"}}
            )
        example = next(
            (e for e in state.comparison.examples if e.query_id == payload.query_id), None
        )
        if example is None:
            return JSONResponse(status_code=404, content={"error": {"code": "unknown_query_id"}})
        return JSONResponse(
            {
                "mode": "saved_development",
                "fixture_id": state.comparison.fixture_id,
                "partition": "development",
                "track": "B",
                "top_k": 10,
                "example": example.model_dump(),
                "sources": state.comparison.sources.model_dump(),
                "pipelines": {k: v.model_dump() for k, v in state.comparison.pipelines.items()},
                "timing_ms": {"replay_lookup": (time.perf_counter() - started) * 1000},
            }
        )

    @app.post("/rerank")
    def rerank(payload: RerankRequest) -> JSONResponse:
        if state.ranking is None:
            return JSONResponse(status_code=503, content={"error": {"code": "ranker_unavailable"}})
        context = state.ranking.context
        candidates: list[tuple[str, str]] = [(payload.locale, pid) for pid in payload.product_ids]
        if not set(candidates).issubset(context.lexical.catalog):
            return JSONResponse(status_code=422, content={"error": {"code": "unknown_candidate"}})
        try:
            ranked = (
                state.serving.rerank(payload.query, candidates)
                if state.serving
                else state.ranking.predict(payload.query, candidates)
            )
        except (PipelineBusy, PipelineUnavailable) as error:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": str(error)}},
                headers={"Retry-After": "1"},
            )
        except MemoryError:
            state.serving = None
            state.ranking = None
            state.index = None
            return JSONResponse(status_code=503, content={"error": {"code": "resource_exhausted"}})
        except (ValueError, IndexError, RuntimeError):
            LOGGER.error(json.dumps({"event": "ranker_failed", "error_type": "feature_contract"}))
            return JSONResponse(
                status_code=500, content={"error": {"code": "feature_contract_failed"}}
            )
        return JSONResponse(
            {
                "mode": "fixed_candidate_learned"
                if state.ranking.ranker.model.trained
                else "fixed_candidate_fixture",
                "learned": state.ranking.ranker.model.trained,
                "pipeline_id": "m4-xgboost-fixed-v1"
                if state.ranking.ranker.model.trained
                else "deterministic-ranker-fixture-v1",
                "model_id": state.ranking.ranker.artifact_id,
                "feature_schema_id": context.provenance.schema_id,
                "candidate_count": len(candidates),
                "results": [
                    {
                        "product_locale": key[0],
                        "product_id": key[1],
                        "title": context.lexical.titles[context.lexical.row_for_key[key]],
                        "score": score,
                    }
                    for key, score in ranked[: payload.top_k]
                ],
            }
        )

    @app.post("/search")
    def search(payload: SearchRequest, request: Request) -> JSONResponse:
        if complete_configured and state.serving is None:
            return JSONResponse(status_code=503, content={"error": {"code": "bundle_unavailable"}})
        if state.serving is not None:
            try:
                result = state.serving.search(payload.query, payload.pipeline, payload.top_k)
                result["request_id"] = request.state.request_id
                metrics.stages(payload.pipeline, result["timing_ms"])
                return JSONResponse(result)
            except (PipelineUnavailable, PipelineBusy) as error:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"code": str(error)}},
                    headers={"Retry-After": "1"},
                )
            except MemoryError:
                state.serving = None
                state.ranking = None
                state.index = None
                LOGGER.error(json.dumps({"event": "serving_unavailable", "error_type": "memory"}))
                return JSONResponse(
                    status_code=503, content={"error": {"code": "resource_exhausted"}}
                )
            except (ValueError, RuntimeError, KeyError, IndexError):
                LOGGER.error(json.dumps({"event": "search_failed", "error_type": "pipeline"}))
                return JSONResponse(status_code=500, content={"error": {"code": "pipeline_failed"}})
        if payload.pipeline != "bm25":
            return JSONResponse(
                status_code=503, content={"error": {"code": "pipeline_unavailable"}}
            )
        if state.index is None:
            return JSONResponse(status_code=503, content={"error": {"code": "index_unavailable"}})
        started = time.perf_counter()
        try:
            hits = state.index.search(payload.query, payload.top_k)
        except (ValueError, IndexError, RuntimeError):
            LOGGER.error(json.dumps({"event": "search_failed", "error_type": "lexical_search"}))
            return JSONResponse(status_code=500, content={"error": {"code": "search_failed"}})
        return JSONResponse(
            {
                "pipeline_id": "bm25-title-lucene-v1",
                "bundle_id": state.index.bundle_id,
                "results": [asdict(hit) for hit in hits],
                "returned": len(hits),
                "timing_ms": {"search": (time.perf_counter() - started) * 1000},
            }
        )

    return app
