"""FAISS-only M5 legacy chunk adapter; deliberately separate from Torch tests."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from atlasrank.artifacts.dense import file_hash
from atlasrank.retrieval.m5_artifacts import build_historical_index, validate_historical_index


def fixture(root: Path, dimensions: int = 2) -> tuple[Path, Path, dict[str, Any]]:
    source = root / "catalog.parquet"
    products = pl.DataFrame(
        {
            "product_locale": ["us"] * 3,
            "product_id": ["a", "b", "c"],
            "product_title": ["first", "second", "third"],
        }
    )
    products.write_parquet(source)
    pins = {
        "catalog_sha256": file_hash(source),
        "catalog_rows": 3,
        "model_weights_sha256": "fixture",
        "dimensions": dimensions,
        "chunk_rows": 2,
    }
    chunks = root / "chunks"
    for start in [0, 2]:
        path = chunks / f"{start:012d}"
        path.mkdir(parents=True)
        frame = products.slice(start, 2)
        frame.write_parquet(path / "products.parquet")
        np.save(path / "vectors.npy", np.eye(dimensions, dtype=np.float32)[: frame.height])
        manifest = {
            "start": start,
            "stop": start + frame.height,
            "rows": frame.height,
            "dimensions": dimensions,
            "dtype": "float32",
            "catalog_sha256": pins["catalog_sha256"],
            "model_weights_sha256": "fixture",
            "vectors_sha256": file_hash(path / "vectors.npy"),
            "products_sha256": file_hash(path / "products.parquet"),
        }
        (path / "manifest.json").write_text(json.dumps(manifest))
    return source, chunks, pins


def test_native_complete_save_load_alignment(tmp_path: Path) -> None:
    source, chunks, pins = fixture(tmp_path)
    out = tmp_path / "index"
    manifest = build_historical_index(source, chunks, out, pins)
    assert manifest["vectors"] == 3
    index, ids = validate_historical_index(out, pins, source=source, chunks=chunks)
    assert ids == [("us", "a"), ("us", "b"), ("us", "c")]
    assert index.ntotal == 3 and index.d == 2
    with pytest.raises(FileExistsError):
        build_historical_index(source, chunks, out, pins)
    path = chunks / "000000000002" / "vectors.npy"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash"):
        validate_historical_index(out, pins, source=source, chunks=chunks)


def test_row_alignment_and_missing_chunk_fail(tmp_path: Path) -> None:
    source, chunks, pins = fixture(tmp_path)
    path = chunks / "000000000002" / "products.parquet"
    pl.read_parquet(path).with_columns(pl.lit("z").alias("product_id")).write_parquet(path)
    mpath = path.parent / "manifest.json"
    m = json.loads(mpath.read_text())
    m["products_sha256"] = file_hash(path)
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="canonical"):
        build_historical_index(source, chunks, tmp_path / "bad", pins)
    mpath.unlink()
    with pytest.raises(ValueError, match="inventory"):
        build_historical_index(source, chunks, tmp_path / "missing", pins)


def test_canonical_chunks_resume_bundle_and_evaluation(tmp_path: Path) -> None:
    from atlasrank.evaluation.m5 import evaluate_rankings
    from atlasrank.retrieval.chunks import encode_chunks
    from atlasrank.retrieval.dense_bundle import build_bundle, load_bundle

    class Encoder:
        recipe: dict[str, Any] = {"dimensions": 2, "batch_size": 2, "kind": "synthetic"}
        calls = 0

        def encode(self, texts: list[str]) -> Any:
            self.calls += 1
            return np.array([[1, 0] if t == "first" else [0, 1] for t in texts], dtype=np.float32)

    source, _, _ = fixture(tmp_path)
    encoder = Encoder()
    job = tmp_path / "canonical-job"
    origin = {"scope": "fixture"}
    encode_chunks(source, job, encoder, origin, chunk_rows=2, wall_seconds=60)
    calls = encoder.calls
    encode_chunks(source, job, encoder, origin, chunk_rows=2, wall_seconds=60)
    assert encoder.calls == calls  # validated chunks never re-encoded
    bundle = tmp_path / "canonical-index"
    build_bundle(job, bundle)
    index, products, _ = load_bundle(bundle, encoder.recipe, expected_origin=origin)
    assert products.height == 3
    from atlasrank.artifacts.dense import product_keys

    development = pl.DataFrame(
        {
            "query_id": [1],
            "query": ["first"],
            "product_locale": ["us"],
            "product_id": ["a"],
            "esci_label": ["E"],
            "partition": ["development"],
        }
    )
    ranked = [key for key, _ in index.search(encoder.encode(["first"]), 3)]
    _, summary = evaluate_rankings(
        development, {1: []}, {1: ranked}, frozenset(product_keys(products))
    )
    assert summary["dense"]["known_exact_recall_at_20"] == 1.0


@pytest.mark.parametrize("field", ["job_id", "chunks"])
def test_wrong_job_bundle_rejected(field: str) -> None:
    from atlasrank.retrieval.m5_artifacts import verify_bundle_job

    job = {"job_id": "correct", "source_sha256": "source", "rows": 3}
    coverage = {"chunk_ids": ["a", "b"]}
    manifest = {**job, "chunks": coverage["chunk_ids"]}
    verify_bundle_job(manifest, job, coverage)
    manifest[field] = "different-compatible-artifact"
    with pytest.raises(ValueError, match="verified canonical job"):
        verify_bundle_job(manifest, job, coverage)


@pytest.mark.parametrize("index_format", ["historical", "rebuilt-historical"])
def test_track_b_tiny_native_command(tmp_path: Path, index_format: str) -> None:
    import argparse

    from atlasrank.artifacts.dense import write_once
    from atlasrank.experiments.m5 import run_track_b

    source, chunks, pins = fixture(tmp_path, dimensions=384)
    index = tmp_path / "index"
    built = build_historical_index(source, chunks, index, pins)
    pins.update(
        model_files={"model.safetensors": "fixture"},
        base_revision="fixture",
        development_sha256="fixture",
        historical_index={"index_sha256": built["index_sha256"], "ids_sha256": built["ids_sha256"]},
    )
    # Reconstruction mode may use a new serialization without changing old artifact pins.
    if index_format == "rebuilt-historical":
        pins["historical_index"]["ids_sha256"] = "different-historical-serialization"
    frame = pl.DataFrame(
        {
            "query_id": [1],
            "query": ["first"],
            "product_locale": ["us"],
            "product_id": ["a"],
            "esci_label": ["E"],
            "partition": ["development"],
            "group_sha256": ["group"],
        }
    )
    queries = tmp_path / "queries"
    queries.mkdir()
    np.save(queries / "vectors.npy", np.eye(384, dtype=np.float32)[:1])
    frame.select("query_id", "group_sha256").write_parquet(queries / "queries.parquet")
    recipe = {
        "kind": "m5-fixed-query-v1",
        "dimensions": 384,
        "batch_size": 32,
        "model_batch_size": 32,
        "max_length": 256,
        "dtype": "float32",
        "preprocessing": "normalize_query:NFKC-casefold-whitespace",
        "padding": "fixed-256",
        "extra_normalization": False,
        "model_files": pins["model_files"],
        "base_revision": "fixture",
        "sentence_transformers": "6.1.0",
        "runtime": {"device": "fixture-only"},
    }
    write_once(
        queries / "manifest.json",
        {
            "recipe": recipe,
            "development_sha256": "fixture",
            "files": {n: file_hash(queries / n) for n in ["vectors.npy", "queries.parquet"]},
        },
    )
    bm25 = tmp_path / "bm25.jsonl"
    bm25.write_text(json.dumps({"query_id": 1, "top_100": []}) + "\n")
    pins["bm25_export_sha256"] = file_hash(bm25)
    args = argparse.Namespace(
        output=tmp_path / "result.json",
        index=index,
        catalog=source,
        chunks=chunks,
        queries=queries,
        bm25=bm25,
        index_format=index_format,
    )
    result = run_track_b(args, pins, frame)
    assert result["summary"]["dense"]["known_exact_recall_at_100"] == 1
    assert result["query_manifest"]["recipe"]["runtime"]["device"] == "fixture-only"
    assert result["index_format"] == index_format
