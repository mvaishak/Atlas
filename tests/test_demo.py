"""Demo contracts exercise the existing app; no model, FAISS or private data."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_search_api import fixture_index

from atlasrank.artifacts.dense import digest, file_hash
from atlasrank.retrieval.bm25 import load
from atlasrank.serving.app import create_app
from atlasrank.serving.comparison import REPO, REVISION


def test_demo_page_is_static_and_separates_live_from_replay() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/demo")
        assert response.status_code == 200
        assert "Live Search" in response.text
        assert "Saved Comparison" in response.text
        assert "not live neural inference" in response.text
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "default-src 'none'" in response.headers["content-security-policy"]


def test_missing_comparison_does_not_change_existing_readiness() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/health/live").json() == {"live": True}
        assert client.get("/health/ready").status_code == 503
        for response in [
            client.get("/comparison"),
            client.post("/comparison", json={"query_id": 1}),
        ]:
            assert response.status_code == 503
            assert response.json() == {"error": {"code": "comparison_unavailable"}}


ATTACK = '<img src=x onerror="document.body.dataset.pwned=1">'


def demo_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    index = tmp_path / "index"
    fixture_index(index)
    lexical = load(index).bundle_id
    recipe = {"repo": REPO, "revision": REVISION}
    provenance = {
        "lexical_artifact_id": lexical,
        "dense_artifact_id": "fixture-dense",
        "product_recipe": recipe,
        "query_recipe": recipe,
        "product_recipe_id": digest(recipe),
        "recipe_id": digest(recipe),
        "source": {"development_sha256": "d" * 64},
    }
    hit = {
        "rank": 1,
        "product_locale": "us",
        "product_id": "a",
        "title": ATTACK,
        "score": 3.1250000000000004,
        "judgment": "E",
    }
    payload = {
        "schema_version": 1,
        "partition": "development",
        "track": "B",
        "mode": "saved_development",
        "evaluation_contract": "v0.3",
        "retriever_k": 100,
        "rrf_constant": 60,
        "sources": {
            "report_sha256": "a" * 64,
            "per_query_sha256": "b" * 64,
            "title_projection_sha256": "c" * 64,
            "development_sha256": "d" * 64,
            "evaluation_contract_sha256": "e" * 64,
            "provenance": provenance,
        },
        "pipelines": {
            "bm25": {
                "pipeline_id": "bm25-title-lucene-v1",
                "bundle_id": lexical,
                "score_kind": "bm25",
                "historical_search_ms": None,
            },
            "hybrid": {
                "pipeline_id": "rrf60-m4-pretrained-v1",
                "bundle_id": f"rrf60:{lexical}:fixture-dense",
                "score_kind": "rrf60",
                "historical_search_ms": None,
            },
        },
        "examples": [
            {
                "query_id": i,
                "query": ATTACK if i == 1 else f"query {i}",
                "group_sha256": str(i) * 64,
                "rankings": {
                    "bm25": [] if i == 5 else [hit],
                    "hybrid": [{**hit, "score": 1 / 61, "judgment": "unknown"}],
                },
            }
            for i in range(1, 6)
        ],
    }
    payload["fixture_id"] = digest(payload)
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload))
    return index, path, payload


def test_replay_exact_values_order_identity_and_null_timing(tmp_path: Path) -> None:
    index, path, source = demo_fixture(tmp_path)
    with TestClient(create_app(index, path, file_hash(path))) as client:
        listing = client.get("/comparison").json()
        assert [q["query_id"] for q in listing["examples"]] == [1, 2, 3, 4, 5]
        for qid in [1, 5]:
            response = client.post("/comparison", json={"query_id": qid})
            assert response.status_code == 200
            data = response.json()
            assert data["example"] == source["examples"][qid - 1]
            assert data["pipelines"] == source["pipelines"]
            assert data["sources"] == source["sources"]
            assert data["mode"] == "saved_development"
            assert data["timing_ms"]["replay_lookup"] >= 0
            assert data["pipelines"]["bm25"]["historical_search_ms"] is None
        assert (
            client.post("/comparison", json={"query_id": 1}).json()["example"]["rankings"]["bm25"][
                0
            ]["score"]
            == 3.1250000000000004
        )
        # Static HTML never interpolates query/title data into markup.
        assert ATTACK not in client.get("/demo").text
        assert client.post("/comparison", json={"query_id": 999}).json() == {
            "error": {"code": "unknown_query_id"}
        }
        expected = load(index).search("apple", 10)
        actual = client.post("/search", json={"query": "apple"}).json()
        assert [(h["product_id"], h["score"]) for h in actual["results"]] == [
            (h.product_id, h.score) for h in expected
        ]
        assert client.get("/health/ready").json() == {
            "ready": True,
            "bundle_id": load(index).bundle_id,
        }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query_id": "1"},
        {"query_id": True},
        {"query_id": -1},
        {"query_id": 1.5},
        {"query_id": 1, "query": "extra"},
    ],
)
def test_comparison_invalid_input(payload: dict[str, Any]) -> None:
    with TestClient(create_app()) as client:
        r = client.post("/comparison", json=payload)
        assert r.status_code == 422 and r.json() == {"error": {"code": "invalid_request"}}


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "missing",
        "bundle",
        "model",
        "partition",
        "duplicate",
        "rank",
        "timing",
        "judgment",
        "score",
    ],
)
def test_bad_comparison_cannot_break_live_search(tmp_path: Path, change: str) -> None:
    index, path, payload = demo_fixture(tmp_path)
    if change == "bundle":
        payload["sources"]["provenance"]["lexical_artifact_id"] = "wrong"
    if change == "model":
        payload["sources"]["provenance"]["query_recipe"]["revision"] = "m5"
    if change == "partition":
        payload["partition"] = "test"
    if change == "duplicate":
        payload["examples"][1]["query_id"] = 1
    if change == "rank":
        payload["examples"][0]["rankings"]["bm25"][0]["rank"] = 2
    if change == "timing":
        payload["pipelines"]["hybrid"]["historical_search_ms"] = 2.0
    if change == "judgment":
        payload["examples"][0]["rankings"]["hybrid"][0]["judgment"] = "irrelevant"
    if change == "score":
        payload["examples"][0]["rankings"]["hybrid"][0]["score"] = float("nan")
    if change != "score":
        payload["fixture_id"] = digest({k: v for k, v in payload.items() if k != "fixture_id"})
    path.write_text(json.dumps(payload))
    sha = "0" * 64 if change == "hash" else file_hash(path)
    if change == "missing":
        path.unlink()
    with TestClient(create_app(index, path, sha)) as client:
        assert client.get("/comparison").status_code == 503
        assert client.get("/health/ready").status_code == 200
        assert client.post("/search", json={"query": "apple"}).status_code == 200


def test_demo_theme_stylesheet_is_local_and_available_without_index() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/demo")
        stylesheet = client.get("/demo/tokens.css")
        assert stylesheet.status_code == 200
        assert stylesheet.headers["content-type"].startswith("text/css")
        assert 'href="/demo/tokens.css"' in page.text
        assert "style-src 'self'" in page.headers["content-security-policy"]
