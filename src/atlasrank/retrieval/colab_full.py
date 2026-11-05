"""Canonical catalog coverage and persistent full-job engine, usable on tiny fixtures."""

import json
import time
from collections.abc import Callable
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from numpy.typing import NDArray

from atlasrank.artifacts.dense import digest, file_hash, product_keys
from atlasrank.retrieval.chunks import Encoder, completed_chunks, encode_chunks
from atlasrank.retrieval.colab_pilot import persist_metadata, restore_job
from atlasrank.retrieval.colab_runner import size
from atlasrank.retrieval.colab_transport import publish

Event = Callable[[dict[str, Any]], None]


class ObservedEncoder:
    """Retain encoder math/recipe; forward existing component timers to the run log."""

    def __init__(self, inner: Encoder, event: Event) -> None:
        self.inner = inner
        self.recipe = inner.recipe
        self.event = event

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        values = self.inner.encode(texts)
        timings = getattr(self.inner, "timings", [])
        if timings:
            self.event({"stage": "encoder_components", **timings[-1]})
        return values


def catalog_contract(source: Path, rows: int, chunk_rows: int = 1024) -> dict[str, Any]:
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow.names != ["product_locale", "product_id", "product_title"]:
        raise ValueError("Only canonical identities and title columns may be exported")
    chunks = []
    start = 0
    previous = ("", "")
    for batch in parquet.iter_batches(batch_size=chunk_rows):
        frame = pl.from_arrow(batch)
        assert isinstance(frame, pl.DataFrame)
        keys = product_keys(frame)
        if keys[0] <= previous:
            raise ValueError("Catalog order differs")
        stop = start + len(keys)
        chunks.append({"start": start, "stop": stop, "content_sha256": digest(frame.rows())})
        previous = keys[-1]
        start = stop
    if start != rows or rows < 1:
        raise ValueError("Full catalog coverage differs")
    return {
        "schema_version": 1,
        "rows": rows,
        "chunk_rows": chunk_rows,
        "catalog_sha256": file_hash(source),
        "chunks": chunks,
    }


def verify_coverage(
    job: Path,
    source: Path,
    contract: dict[str, Any],
    recipe: dict[str, Any],
    origin: dict[str, Any],
    checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    metadata = json.loads((job / "JOB.json").read_text())
    if (
        metadata["recipe"] != recipe
        or metadata["origin"] != origin
        or metadata["source_sha256"] != file_hash(source)
        or metadata["source_sha256"] != contract["catalog_sha256"]
        or metadata["chunk_rows"] != contract["chunk_rows"]
        or metadata["rows"] != contract["rows"]
    ):
        raise ValueError("Full job lineage/coverage differs")
    total = 0
    ids = []
    for expected, item in zip_longest(contract["chunks"], completed_chunks(job)):
        if expected is None or item is None:
            raise ValueError("Full chunk inventory differs")
        if checkpoint:
            checkpoint()
        manifest, products, _ = item
        if (
            manifest["start"] != expected["start"]
            or manifest["stop"] != expected["stop"]
            or digest(products.rows()) != expected["content_sha256"]
        ):
            raise ValueError("Full chunk identities/titles/boundaries differ")
        total += products.height
        ids.append(manifest["artifact_id"])
    if total != contract["rows"]:
        raise ValueError("Incomplete full row coverage")
    return {"rows": total, "chunks": len(ids), "chunk_ids": ids, "job_id": metadata["job_id"]}


def encode_persisted(
    source: Path,
    workspace: Path,
    encoder: Encoder,
    origin: dict[str, Any],
    store: Path,
    *,
    chunk_rows: int = 1024,
    wall_seconds: float = 18000,
    checkpoint: Callable[[], None] | None = None,
    event: Event | None = None,
) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    staging = workspace / "staging"
    staging.mkdir(exist_ok=True)
    job = workspace / "job"
    if not job.exists():
        restore_job(store, job, staging, 4 * 1024**3, checkpoint)
    report_event = event or (lambda record: None)
    rows = pq.ParquetFile(source).metadata.num_rows
    contract = catalog_contract(source, rows, chunk_rows)

    def commit(chunk: Path) -> None:
        if checkpoint:
            checkpoint()
        start = time.perf_counter()
        if size(store) + size(chunk) + 1024**2 > 4 * 1024**3:
            raise RuntimeError("Insufficient persistent checkpoint headroom; preserve history")
        persist_metadata(job, store, staging, 4 * 1024**3)
        publish(chunk, store / "chunks", chunk.name, 4 * 1024**3)
        report_event(
            {
                "stage": "checkpoint_publication",
                "chunk_start": int(chunk.name),
                "seconds": time.perf_counter() - start,
            }
        )
        if checkpoint:
            checkpoint()

    try:
        encode_chunks(
            source,
            job,
            ObservedEncoder(encoder, report_event),
            origin,
            chunk_rows=chunk_rows,
            wall_seconds=wall_seconds,
            budget_root=workspace,
            checkpoint=checkpoint,
            on_chunk=commit,
            event=report_event,
        )
    finally:
        persist_metadata(job, store, staging, 4 * 1024**3)
    result = verify_coverage(job, source, contract, encoder.recipe, origin, checkpoint)
    report_event({"stage": "coverage_verified", "rows": result["rows"], "chunks": result["chunks"]})
    return result
