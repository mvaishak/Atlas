"""HTTP search uses the actual adapter; no network service or dataset required."""

import json
import logging
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from atlasrank.retrieval.bm25 import build, load
from atlasrank.serving.app import create_app


def fixture_index(path: Path) -> None:
    build(
        pl.DataFrame(
            {
                "product_locale": ["us", "us"],
                "product_id": ["a", "b"],
                "product_title": ["red apple", "blue berry"],
            }
        ),
        path,
        {"kind": "synthetic"},
    )


def test_http_offline_parity_and_query_not_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "index"
    fixture_index(path)
    expected = load(path).search(" ＲＥＤ apple ", 10)
    caplog.set_level(logging.INFO, logger="atlasrank.http")
    with TestClient(create_app(path)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        response = client.post("/search", json={"query": " ＲＥＤ apple ", "top_k": 10})
        assert response.status_code == 200
        payload = response.json()
        assert [(r["product_locale"], r["product_id"], r["score"]) for r in payload["results"]] == [
            (hit.product_locale, hit.product_id, hit.score) for hit in expected
        ]
        assert payload["timing_ms"]["search"] >= 0
        assert payload["bundle_id"] == load(path).bundle_id
        assert response.headers["X-Request-ID"]
        assert client.post("/search", json={"query": "unknownzz"}).json()["results"] == []
    events = [json.loads(r.message) for r in caplog.records if r.name == "atlasrank.http"]
    assert events and all("query" not in event and "product_id" not in event for event in events)
    assert "ＲＥＤ" not in " ".join(r.message for r in caplog.records if r.name == "atlasrank.http")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "!!!"},
        {"query": "a" * 513},
        {"query": "a", "top_k": 0},
        {"query": "a", "top_k": 101},
        {"query": "a", "top_k": True},
        {"query": "a", "top_k": "10"},
        {"query": "a", "locale": "jp"},
        {"query": "a", "extra": 1},
        {"query": "a " * 129},
        {"query": 123},
    ],
)
def test_invalid_requests_have_explicit_safe_error(payload: dict[str, object]) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/search", json=payload)
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_request"}}


def test_missing_and_corrupt_indexes_stay_unready(tmp_path: Path) -> None:
    paths = [tmp_path / "missing", tmp_path / "corrupt"]
    fixture_index(paths[1])
    (paths[1] / "COMPLETE.json").write_bytes(b"\xff")
    for path in paths:
        with TestClient(create_app(path)) as client:
            assert client.get("/health/live").status_code == 200
            assert client.get("/health/ready").status_code == 503
            response = client.post("/search", json={"query": "apple"})
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "index_unavailable"


def test_internal_search_failure_has_no_query_or_exception_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from atlasrank.retrieval.bm25 import LexicalIndex

    path = tmp_path / "index"
    fixture_index(path)

    def fail(self: LexicalIndex, query: str, k: int = 10) -> list[object]:
        raise RuntimeError(f"private failure containing {query}")

    monkeypatch.setattr(LexicalIndex, "search", fail)
    with TestClient(create_app(path)) as client:
        response = client.post("/search", json={"query": "sensitivequerymarker"})
        assert response.status_code == 500
        assert response.json() == {"error": {"code": "search_failed"}}
    assert "sensitivequerymarker" not in caplog.text
