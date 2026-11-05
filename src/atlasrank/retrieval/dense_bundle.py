"""Build and load fully aligned dense bundles from completed immutable chunk jobs."""

import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from atlasrank.artifacts.dense import InvalidDense, digest, product_keys, seal, validate
from atlasrank.data.common import check_disk
from atlasrank.retrieval.chunks import account_work, completed_chunks
from atlasrank.retrieval.dense import DenseIndex
from atlasrank.retrieval.encoder import validate_vectors

BUNDLE_FILES = {"products.parquet", "vectors.npy", "index.faiss"}


def build_bundle(
    job_directory: Path,
    output: Path,
    *,
    budget_root: Path | None = None,
    checkpoint: Callable[[], None] | None = None,
    accounting_directory: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve existing dense bundles")
    budget_root = budget_root or output.parent
    with account_work(accounting_directory or job_directory, "bundle") as tick:
        job = json.loads((job_directory / "JOB.json").read_text())
        dims, count = job["recipe"]["dimensions"], job["rows"]
        check_disk(budget_root, count * dims * 8 + 64 * 1024**2)
        output.mkdir(parents=True)
        vectors = np.lib.format.open_memmap(
            output / "vectors.npy", mode="w+", dtype=np.float32, shape=(count, dims)
        )
        index = faiss.IndexFlatIP(dims)
        faiss.omp_set_num_threads(1)
        chunk_ids = []
        writer: Any = None
        try:
            for manifest, products, chunk in completed_chunks(job_directory):
                tick()
                if checkpoint is not None:
                    checkpoint()
                start, stop = manifest["start"], manifest["stop"]
                vectors[start:stop] = chunk
                index.add(np.ascontiguousarray(chunk))
                table = products.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        output / "products.parquet", table.schema, compression="zstd"
                    )
                writer.write_table(table, row_group_size=job["chunk_rows"])
                chunk_ids.append(manifest["artifact_id"])
                check_disk(budget_root)
        finally:
            if writer is not None:
                writer.close()
        vectors.flush()
        del vectors
        if index.ntotal != count:
            raise InvalidDense("Bundle index count differs")
        faiss.write_index(index, str(output / "index.faiss"))
        tick()
        if checkpoint is not None:
            checkpoint()
        return seal(
            output,
            {
                "schema_version": 1,
                "kind": "atlasrank.dense",
                "dimensions": dims,
                "rows": count,
                "dtype": "float32",
                "metric": "normalized_inner_product",
                "index_type": "IndexFlatIP",
                "faiss_version": importlib.metadata.version("faiss-cpu"),
                "numpy_version": np.__version__,
                "job_id": job["job_id"],
                "recipe": job["recipe"],
                "recipe_id": digest(job["recipe"]),
                "origin": job["origin"],
                "source_sha256": job["source_sha256"],
                "chunks": chunk_ids,
            },
            BUNDLE_FILES,
        )


def load_bundle(
    directory: Path,
    expected_recipe: dict[str, Any],
    *,
    expected_origin: dict[str, Any] | None = None,
) -> tuple[DenseIndex, pl.DataFrame, dict[str, Any]]:
    try:
        manifest = validate(directory, "atlasrank.dense", BUNDLE_FILES)
        if (
            manifest["recipe_id"] != digest(expected_recipe)
            or manifest["recipe"] != expected_recipe
        ):
            raise InvalidDense("Encoder/tokenizer/backend recipe differs")
        if expected_origin is not None and manifest["origin"] != expected_origin:
            raise InvalidDense("Dense source origin differs")
        if (
            manifest["faiss_version"] != importlib.metadata.version("faiss-cpu")
            or manifest["numpy_version"] != np.__version__
        ):
            raise InvalidDense("Dense numerical dependency versions differ")
        if (
            manifest["index_type"] != "IndexFlatIP"
            or manifest["metric"] != "normalized_inner_product"
            or manifest["dtype"] != "float32"
        ):
            raise InvalidDense("Incompatible dense scoring configuration")
        products = pl.read_parquet(directory / "products.parquet")
        keys = product_keys(products)
        vectors = np.load(directory / "vectors.npy", allow_pickle=False, mmap_mode="r")
        if (
            vectors.shape != (len(keys), expected_recipe["dimensions"])
            or manifest["rows"] != len(keys)
            or manifest["dimensions"] != expected_recipe["dimensions"]
        ):
            raise InvalidDense("Dense product/vector dimensions differ")
        index = faiss.read_index(str(directory / "index.faiss"))
        if (
            not isinstance(index, faiss.IndexFlatIP)
            or index.d != manifest["dimensions"]
            or index.ntotal != len(keys)
            or index.metric_type != faiss.METRIC_INNER_PRODUCT
        ):
            raise InvalidDense("Dense native index type/count/metric differs")
        for start in range(0, len(keys), 1024):
            chunk = vectors[start : start + 1024]
            validate_vectors(chunk, manifest["dimensions"])
            if not np.array_equal(index.reconstruct_n(start, len(chunk)), chunk):
                raise InvalidDense("Native FAISS/vector row alignment differs")
        return DenseIndex.from_backend(keys, index), products, manifest
    except (
        OSError,
        EOFError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        pl.exceptions.PolarsError,
    ) as error:
        raise InvalidDense("Invalid or incompatible dense bundle") from error
