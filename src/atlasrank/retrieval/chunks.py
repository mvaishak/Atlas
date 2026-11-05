"""Canonical, bounded embedding chunks with locked publication and resumable accounting."""

import fcntl
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from numpy.typing import NDArray

from atlasrank.artifacts.dense import (
    InvalidDense,
    digest,
    file_hash,
    product_keys,
    seal,
    validate,
    write_once,
)
from atlasrank.data.common import check_disk, check_memory
from atlasrank.retrieval.encoder import validate_vectors

CHUNK_FILES = {"products.parquet", "vectors.npy"}


class Encoder(Protocol):
    recipe: dict[str, Any]

    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


def canonical_catalog(source: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Preserve the existing canonical catalog")
    products = pl.read_parquet(
        source, columns=["product_locale", "product_id", "product_title"]
    ).sort("product_locale", "product_id")
    product_keys(products)
    products.write_parquet(output, compression="zstd", row_group_size=1024)
    return output


def validate_chunk(
    path: Path, job: dict[str, Any], expected: pl.DataFrame | None = None
) -> tuple[dict[str, Any], pl.DataFrame, NDArray[np.float32]]:
    try:
        manifest = validate(path, "atlasrank.embedding_chunk", CHUNK_FILES)
        products = pl.read_parquet(path / "products.parquet")
        product_keys(products)
        vectors = np.load(path / "vectors.npy", allow_pickle=False, mmap_mode="r")
        validate_vectors(vectors, job["recipe"]["dimensions"])
        if (
            manifest["job_id"] != job["job_id"]
            or manifest["stop"] - manifest["start"] != products.height
            or len(vectors) != products.height
        ):
            raise InvalidDense("Chunk range/identity/vector alignment differs")
        if expected is not None and not products.equals(expected):
            raise InvalidDense("Chunk products/titles differ from expected source rows")
        return manifest, products, vectors
    except (OSError, EOFError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as error:
        raise InvalidDense("Invalid embedding chunk") from error


def completed_chunks(
    directory: Path,
) -> Iterator[tuple[dict[str, Any], pl.DataFrame, NDArray[np.float32]]]:
    job = json.loads((directory / "JOB.json").read_text())
    payload = {k: v for k, v in job.items() if k != "job_id"}
    if digest(payload) != job["job_id"]:
        raise InvalidDense("Job metadata identity differs")
    start = 0
    ids = []
    previous = ("", "")
    for path in sorted((directory / "chunks").iterdir()):
        manifest, products, vectors = validate_chunk(path, job)
        keys = product_keys(products)
        if manifest["start"] != start or path.name != f"{start:012d}" or keys[0] <= previous:
            raise InvalidDense("Chunk gap, overlap or identity ordering differs")
        start = manifest["stop"]
        previous = keys[-1]
        ids.append(manifest["artifact_id"])
        yield manifest, products, vectors
    complete = json.loads((directory / "JOB_COMPLETE.json").read_text())
    if start != job["rows"] or complete != {"job_id": job["job_id"], "chunks": ids}:
        raise InvalidDense("Incomplete embedding job")


def accounted_seconds(directory: Path) -> float:
    total = 0.0
    for start in sorted((directory / "attempts").glob("*.start.json")):
        initial = json.loads(start.read_text())
        end = start.with_name(start.name.replace(".start.json", ".end.json"))
        recovery = start.with_name(start.name.replace(".start.json", ".recovered.json"))
        if end.exists():
            charge = float(json.loads(end.read_text())["elapsed_seconds"])
        elif recovery.exists():
            charge = float(json.loads(recovery.read_text())["charged_seconds"])
        else:
            # The exclusive lock proves the previous writer is gone. Include
            # downtime conservatively when recovering an unclosed hard-kill attempt.
            elapsed = datetime.now(UTC).timestamp() - float(initial["utc_start"])
            charge = (
                initial["reserved_seconds"]
                if elapsed < 0
                else min(initial["reserved_seconds"], elapsed)
            )
            write_once(
                recovery,
                {
                    "charged_seconds": charge,
                    "policy": "conservative elapsed wall time including downtime",
                },
            )
        if charge < 0 or not np.isfinite(charge):
            raise InvalidDense("Invalid cumulative attempt accounting")
        total += charge
    return total


@contextmanager
def account_work(directory: Path, phase: str) -> Iterator[Callable[[], None]]:
    """Use the embedding job's same cumulative budget for bundle/evaluation phases."""
    with (directory / ".writer.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = json.loads((directory / "JOB.json").read_text())
        charged = accounted_seconds(directory)
        remaining = float(job["wall_seconds"]) - charged
        if remaining <= 0:
            raise RuntimeError("Cumulative dense run wall budget exhausted")
        started = time.monotonic()
        attempt = uuid.uuid4().hex
        write_once(
            directory / "attempts" / f"{attempt}.start.json",
            {
                "utc_start": datetime.now(UTC).timestamp(),
                "reserved_seconds": remaining,
                "phase": phase,
            },
        )

        def tick() -> None:
            if time.monotonic() - started >= remaining:
                raise RuntimeError("Cumulative dense run wall budget exhausted")
            check_memory()

        status = "failed"
        try:
            yield tick
            tick()
            status = "passed"
        finally:
            write_once(
                directory / "attempts" / f"{attempt}.end.json",
                {"elapsed_seconds": time.monotonic() - started, "phase": phase, "status": status},
            )


def encode_chunks(
    source: Path,
    output: Path,
    encoder: Encoder,
    origin: dict[str, Any],
    *,
    chunk_rows: int = 1024,
    wall_seconds: float = 43200,
    budget_root: Path | None = None,
    checkpoint: Callable[[], None] | None = None,
    on_chunk: Callable[[Path], None] | None = None,
    event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    function_started = time.monotonic()
    batch_size = int(encoder.recipe["batch_size"])
    if not 1 <= chunk_rows <= 4096 or batch_size < 1 or chunk_rows % batch_size:
        raise ValueError("Chunk boundaries must align to fixed encoding batches")
    if wall_seconds <= 0 or not np.isfinite(wall_seconds):
        raise ValueError("Expected a finite cumulative wall budget")
    budget_root = budget_root or output.parent
    check_disk(budget_root)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Embedding job already has an active writer") from error
        for name in ("chunks", "partials", "attempts"):
            (output / name).mkdir(exist_ok=True)
        payload = {
            "schema_version": 1,
            "source_sha256": file_hash(source),
            "rows": pq.ParquetFile(source).metadata.num_rows,
            "recipe": json.loads(json.dumps(encoder.recipe, allow_nan=False)),
            "origin": json.loads(json.dumps(origin, allow_nan=False)),
            "chunk_rows": chunk_rows,
            "wall_seconds": wall_seconds,
        }
        job = {**payload, "job_id": digest(payload)}
        if job["rows"] < 1:
            raise ValueError("Cannot encode an empty catalog")
        if (output / "JOB.json").exists():
            if json.loads((output / "JOB.json").read_text()) != job:
                raise InvalidDense("Resume source/recipe/origin/budget differs; preserve job")
        else:
            write_once(output / "JOB.json", job)
        charged = accounted_seconds(output)
        if charged >= wall_seconds:
            raise RuntimeError("Cumulative embedding wall budget exhausted")
        attempt = uuid.uuid4().hex
        started = function_started
        write_once(
            output / "attempts" / f"{attempt}.start.json",
            {
                "utc_start": datetime.now(UTC).timestamp(),
                "reserved_seconds": wall_seconds - charged,
            },
        )
        status = "failed"
        new_chunks = 0
        try:
            start = 0
            ids = []
            previous = ("", "")
            seen = set()
            for batch in pq.ParquetFile(source).iter_batches(
                batch_size=chunk_rows, columns=["product_locale", "product_id", "product_title"]
            ):
                if checkpoint is not None:
                    checkpoint()
                products = pl.from_arrow(batch)
                assert isinstance(products, pl.DataFrame)
                keys = product_keys(products)
                if keys[0] <= previous:
                    raise InvalidDense("Source catalog isn't globally canonical")
                previous = keys[-1]
                target = output / "chunks" / f"{start:012d}"
                seen.add(target.name)
                stop = start + products.height
                if target.exists():
                    validation_started = time.perf_counter()
                    manifest, _, _ = validate_chunk(target, job, products)
                    if manifest["start"] != start or manifest["stop"] != stop:
                        raise InvalidDense("Resume chunk boundaries differ")
                    if event:
                        event(
                            {
                                "stage": "chunk_reused",
                                "start": start,
                                "stop": stop,
                                "seconds": time.perf_counter() - validation_started,
                            }
                        )
                else:
                    partial = output / "partials" / uuid.uuid4().hex
                    partial.mkdir()
                    products.write_parquet(
                        partial / "products.parquet", compression="zstd", row_group_size=chunk_rows
                    )
                    vectors = np.lib.format.open_memmap(
                        partial / "vectors.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(products.height, int(encoder.recipe["dimensions"])),
                    )
                    titles = products["product_title"].to_list()
                    for offset in range(0, products.height, batch_size):
                        if checkpoint is not None:
                            checkpoint()
                        if charged + time.monotonic() - started >= wall_seconds:
                            raise RuntimeError("Cumulative embedding wall budget exhausted")
                        if encoder.recipe != job["recipe"]:
                            raise InvalidDense("Encoder recipe changed during encoding")
                        batch_started = time.perf_counter()
                        encoded = encoder.encode(titles[offset : offset + batch_size])
                        if event:
                            event(
                                {
                                    "stage": "batch_encoding",
                                    "start": start + offset,
                                    "rows": len(encoded),
                                    "seconds": time.perf_counter() - batch_started,
                                }
                            )
                        if encoder.recipe != job["recipe"]:
                            raise InvalidDense("Encoder recipe changed during encoding")
                        validate_vectors(encoded, int(encoder.recipe["dimensions"]))
                        if len(encoded) != len(titles[offset : offset + batch_size]):
                            raise InvalidDense("Encoder changed product row count")
                        vectors[offset : offset + len(encoded)] = encoded
                        resource_started = time.perf_counter()
                        check_memory()
                        check_disk(budget_root)
                        if event:
                            event(
                                {
                                    "stage": "resource_checks",
                                    "start": start + offset,
                                    "seconds": time.perf_counter() - resource_started,
                                }
                            )
                    publication_started = time.perf_counter()
                    vectors.flush()
                    del vectors
                    manifest = seal(
                        partial,
                        {
                            "schema_version": 1,
                            "kind": "atlasrank.embedding_chunk",
                            "job_id": job["job_id"],
                            "start": start,
                            "stop": stop,
                        },
                        CHUNK_FILES,
                    )
                    validate_chunk(partial, job, products)
                    if target.exists():
                        raise InvalidDense("Chunk publication would overwrite existing data")
                    os.rename(partial, target)
                    descriptor = os.open(output / "chunks", os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    new_chunks += 1
                    if event:
                        event(
                            {
                                "stage": "local_chunk_publication",
                                "start": start,
                                "stop": stop,
                                "seconds": time.perf_counter() - publication_started,
                            }
                        )
                if on_chunk is not None:
                    on_chunk(target)
                ids.append(manifest["artifact_id"])
                start = stop
            if start != job["rows"] or {p.name for p in (output / "chunks").iterdir()} != seen:
                raise InvalidDense("Unexpected chunk inventory or row count")
            complete = {"job_id": job["job_id"], "chunks": ids}
            if (output / "JOB_COMPLETE.json").exists():
                if json.loads((output / "JOB_COMPLETE.json").read_text()) != complete:
                    raise InvalidDense("Completed job differs")
            else:
                write_once(output / "JOB_COMPLETE.json", complete)
            status = "passed"
            return {
                "job_id": job["job_id"],
                "new_chunks": new_chunks,
                "chunks": ids,
                "previous_charged_seconds": charged,
            }
        finally:
            write_once(
                output / "attempts" / f"{attempt}.end.json",
                {
                    "status": status,
                    "elapsed_seconds": time.monotonic() - started,
                    "new_chunks": new_chunks,
                },
            )
