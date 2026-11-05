"""Serving descriptors bind relocated components before any native execution."""

import json
import sys
from pathlib import Path

import pytest


def descriptor(tmp_path: Path) -> Path:
    from atlasrank.artifacts.serving_bundle import prepare

    root = tmp_path / "artifacts"
    root.mkdir()
    for name in ("lexical", "dense", "model", "ranker"):
        (root / name).mkdir()
        (root / name / ("config.json" if name == "model" else "manifest.json")).write_text("{}")
    (root / "dense-config.json").write_text("{}")
    output = tmp_path / "descriptor"
    prepare(
        root,
        output,
        {
            "lexical": "lexical",
            "dense": "dense",
            "model": "model",
            "ranker": "ranker",
            "dense_config": "dense-config.json",
        },
        ["bm25", "dense_m4", "rrf_m4", "m6_xgb_m4"],
        query_recipe_id="a" * 64,
    )
    return output


def test_descriptor_relocation_and_bytes(tmp_path: Path) -> None:
    from atlasrank.artifacts.serving_bundle import preflight

    output = descriptor(tmp_path)
    spec, identity = preflight(output, tmp_path / "artifacts")
    assert spec.pipelines[0] == "bm25" and len(identity) == 64
    (tmp_path / "artifacts/ranker/manifest.json").write_text('{"changed":true}')
    before = set(sys.modules)
    with pytest.raises(ValueError):
        preflight(output, tmp_path / "artifacts")
    assert not {"torch", "xgboost", "faiss"}.intersection(set(sys.modules) - before)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "dense/../../escape"])
def test_descriptor_rejects_unsafe_component_paths(tmp_path: Path, path: str) -> None:
    from atlasrank.artifacts.serving_bundle import prepare

    with pytest.raises(ValueError):
        prepare(tmp_path, tmp_path / "out", {"lexical": path}, ["bm25"])


def test_missing_symlink_and_corrupt_descriptor(tmp_path: Path) -> None:
    from atlasrank.artifacts.serving_bundle import preflight

    output = descriptor(tmp_path)
    target = tmp_path / "artifacts/dense/manifest.json"
    target.unlink()
    target.symlink_to(tmp_path / "artifacts/lexical/manifest.json")
    with pytest.raises(ValueError):
        preflight(output, tmp_path / "artifacts")
    (output / "spec.json").chmod(0o644)
    (output / "spec.json").write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError):
        preflight(output, tmp_path / "artifacts")


def test_actual_bm25_descriptor_startup(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from test_ranking_features import runtime

    from atlasrank.artifacts.serving_bundle import prepare
    from atlasrank.serving.app import create_app

    context = runtime(tmp_path)
    output = tmp_path / "serving"
    prepare(tmp_path, output, {"lexical": "lexical"}, ["bm25"])
    with TestClient(create_app(serving_bundle=output, artifact_root=tmp_path)) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post("/search", json={"query": "apple"})
        assert response.status_code == 200
        assert response.json()["results"][0]["product_id"] == "a"
        assert context.lexical.bundle_id


def test_neural_descriptor_requires_separate_query_recipe_pin() -> None:
    from atlasrank.artifacts.serving_bundle import ServingSpec

    values = {
        "lexical": "lexical",
        "dense": "dense",
        "model": "model",
        "dense_config": "config.json",
    }
    with pytest.raises(ValueError, match="query recipe"):
        ServingSpec(
            components=values, pins={k: "a" * 64 for k in values}, pipelines=["bm25", "rrf_m4"]
        )
