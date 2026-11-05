"""Read-only validation of historical M5 chunks and native-only index rebuilding.

Historical chunks are not M4 sealed chunks. Never relabel or overwrite them.
New encoding uses retrieval.chunks; this adapter only consumes the old format.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from numpy.typing import NDArray

from atlasrank.artifacts.dense import file_hash, product_keys, write_once
from atlasrank.retrieval.m5_encoder import verify_file


def historical_chunks(
    source: Path, chunks: Path, pins: dict[str, Any]
) -> Iterator[tuple[int, pl.DataFrame, NDArray[np.float32]]]:
    verify_file(source, pins["catalog_sha256"])
    rows, size, dims = pins["catalog_rows"], pins["chunk_rows"], pins["dimensions"]
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != rows:
        raise ValueError("Catalog row count differs")
    expected_names = {f"{i:012d}" for i in range(0, rows, size)}
    if chunks.is_symlink() or {p.name for p in chunks.iterdir()} != expected_names:
        raise ValueError("Chunk coverage differs")
    previous = ("", "")
    offset = 0
    for batch in parquet.iter_batches(batch_size=size):
        expected = pl.from_arrow(batch)
        assert isinstance(expected, pl.DataFrame)
        keys = product_keys(expected)
        if keys[0] <= previous:
            raise ValueError("Noncanonical catalog order")
        previous = keys[-1]
        path = chunks / f"{offset:012d}"
        if path.is_symlink() or {p.name for p in path.iterdir()} != {
            "manifest.json",
            "products.parquet",
            "vectors.npy",
        }:
            raise ValueError("Historical chunk inventory differs")
        m = json.loads((path / "manifest.json").read_text())
        for name in ["catalog_sha256", "model_weights_sha256", "dimensions"]:
            if m[name] != pins[name]:
                raise ValueError("Historical chunk provenance differs")
        if (
            m["start"] != offset
            or m["stop"] != offset + expected.height
            or m["rows"] != expected.height
            or m["dtype"] != "float32"
        ):
            raise ValueError("Chunk boundaries differ")
        verify_file(path / "products.parquet", m["products_sha256"])
        verify_file(path / "vectors.npy", m["vectors_sha256"])
        products = pl.read_parquet(path / "products.parquet")
        if not products.equals(expected):
            raise ValueError("Chunk products differ from canonical rows/titles")
        vectors = np.load(path / "vectors.npy", allow_pickle=False, mmap_mode="r")
        if (
            vectors.dtype != np.float32
            or vectors.shape != (expected.height, dims)
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4, rtol=0)
        ):
            raise ValueError("Invalid historical vectors")
        yield offset, products, vectors
        offset += expected.height
    if offset != rows:
        raise ValueError("Incomplete canonical coverage")


def build_historical_index(
    source: Path, chunks: Path, output: Path, pins: dict[str, Any]
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve existing indexes, including partial builds")
    output.mkdir(parents=True)
    index = faiss.IndexFlatIP(pins["dimensions"])
    faiss.omp_set_num_threads(4)
    writer: Any = None
    try:
        for start, products, vectors in historical_chunks(source, chunks, pins):
            if index.ntotal != start:
                raise ValueError("Vector/identity offset differs")
            index.add(np.ascontiguousarray(vectors))
            table = products.select("product_locale", "product_id").to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output / "ids.parquet", table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if index.ntotal != pins["catalog_rows"]:
        raise ValueError("Index incomplete")
    faiss.write_index(index, str(output / "index.faiss"))
    manifest = {
        "schema_version": 1,
        "catalog_sha256": pins["catalog_sha256"],
        "model_weights_sha256": pins["model_weights_sha256"],
        "dimensions": pins["dimensions"],
        "vectors": pins["catalog_rows"],
        "index_type": "IndexFlatIP",
        "faiss_version": faiss.__version__,
        "numpy_version": np.__version__,
        "index_sha256": file_hash(output / "index.faiss"),
        "ids_sha256": file_hash(output / "ids.parquet"),
    }
    write_once(output / "INDEX_MANIFEST.json", manifest)
    validate_historical_index(output, pins, source=source, chunks=chunks)
    return manifest


def validate_historical_index(
    directory: Path, pins: dict[str, Any], *, source: Path, chunks: Path
) -> tuple[Any, list[tuple[str, str]]]:
    if "faiss_version" in pins and faiss.__version__ != pins["faiss_version"]:
        raise ValueError("Native FAISS version differs from recorded M5 runtime")
    manifest = json.loads((directory / "INDEX_MANIFEST.json").read_text())
    for key in ["catalog_sha256", "model_weights_sha256", "dimensions"]:
        if manifest[key] != pins[key]:
            raise ValueError("Index provenance differs")
    if manifest["vectors"] != pins["catalog_rows"] or manifest["index_type"] != "IndexFlatIP":
        raise ValueError("Index metadata differs")
    for name, key in [("index.faiss", "index_sha256"), ("ids.parquet", "ids_sha256")]:
        verify_file(directory / name, manifest[key])
        if key in pins and manifest[key] != pins[key]:
            raise ValueError("Historical index pin differs")
    ids_frame = pl.read_parquet(directory / "ids.parquet")
    if ids_frame.columns != ["product_locale", "product_id"]:
        raise ValueError("ID schema differs")
    ids = list(ids_frame.iter_rows())
    if (
        len(ids) != pins["catalog_rows"]
        or ids != sorted(set(ids))
        or any(loc != "us" or not pid for loc, pid in ids)
    ):
        raise ValueError("ID completeness/order differs")
    index = faiss.read_index(str(directory / "index.faiss"))
    if (
        not isinstance(index, faiss.IndexFlatIP)
        or index.d != pins["dimensions"]
        or index.ntotal != len(ids)
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ValueError("Native index compatibility differs")
    for start, products, vectors in historical_chunks(source, chunks, pins):
        expected_ids = list(products.select("product_locale", "product_id").iter_rows())
        if ids[start : start + len(vectors)] != expected_ids or not np.array_equal(
            index.reconstruct_n(start, len(vectors)), vectors
        ):
            raise ValueError("Native vector-to-ID alignment differs")
    return index, ids


def verify_bundle_job(
    manifest: dict[str, Any], job: dict[str, Any], coverage: dict[str, Any]
) -> None:
    """Bind an independently sealed bundle to the exact verified chunk job."""
    expected = {
        "job_id": job["job_id"],
        "source_sha256": job["source_sha256"],
        "rows": job["rows"],
        "chunks": coverage["chunk_ids"],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Bundle does not belong to the verified canonical job")
