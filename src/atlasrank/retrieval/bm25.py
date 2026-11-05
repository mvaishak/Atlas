"""Pinned BM25S behind deterministic identities, ordering, and serialization."""

import gc
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import polars as pl
from numpy.typing import NDArray

from atlasrank.artifacts.lexical import InvalidIndex, seal, validate
from atlasrank.data.common import check_memory
from atlasrank.evaluation.metrics import ProductKey
from atlasrank.retrieval.analyzer import ANALYZER, analyze

SCORING: dict[str, Any] = {
    "analyzer": ANALYZER,
    "fields": ["product_title"],
    "method": "lucene",
    "k1": 1.5,
    "b": 0.75,
    "dtype": "float32",
    "backend": "numpy",
}
PROBE_QUERIES = ("apple", "wireless headphones", "stainless steel", "atlasrankzzzxqvnoterm")


@dataclass(frozen=True)
class Hit:
    product_locale: str
    product_id: str
    title: str
    score: float


def top_rows(scores: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    """Positive matches only; row IDs are ascending product keys, including ties at k."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("top_k must be an integer from 1 to 100")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Invalid BM25 scores")
    matched = np.flatnonzero(scores > 0)
    if len(matched) > k:
        threshold = np.partition(scores[matched], -k)[-k]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)[: k - len(above)]
        matched = np.concatenate((above, tied))
    return np.asarray(matched[np.lexsort((matched, -scores[matched]))], dtype=np.int64)


class LexicalIndex:
    def __init__(self, backend: Any, products: pl.DataFrame, metadata: dict[str, Any]) -> None:
        self._backend = backend
        self.metadata = metadata
        self.keys: list[ProductKey] = [
            (str(locale), str(pid))
            for locale, pid in products.select("product_locale", "product_id").iter_rows()
        ]
        self.titles: list[str] = products["product_title"].to_list()
        self.row_for_key = {key: i for i, key in enumerate(self.keys)}
        # A real immutable set avoids set.issubset rebuilding the full catalog
        # when the metric helper receives another kind of Set view.
        self.catalog: Set[ProductKey] = frozenset(self.row_for_key)
        if len(self.row_for_key) != len(self.keys) or self.keys != sorted(self.keys):
            raise InvalidIndex("Product identity mapping must be unique and canonical")
        if any(locale != "us" or not pid.strip() for locale, pid in self.keys):
            raise InvalidIndex("Product identities must be nonempty US keys")

    @property
    def bundle_id(self) -> str:
        return str(self.metadata["bundle_id"])

    def score_all(self, query: str) -> NDArray[np.float32]:
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        tokens = [term for term in analyze(query) if term in self._backend.vocab_dict]
        if not tokens:
            return np.zeros(len(self.keys), dtype=np.float32)
        return np.asarray(self._backend.get_scores(tokens), dtype=np.float32)

    def hits_from_scores(self, scores: NDArray[np.float32], k: int) -> list[Hit]:
        if scores.shape != (len(self.keys),):
            raise ValueError("Score/product alignment mismatch")
        return [
            Hit(*self.keys[int(row)], self.titles[int(row)], float(scores[row]))
            for row in top_rows(scores, k)
        ]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.hits_from_scores(self.score_all(query), k)

    def candidate_scores(
        self, scores: NDArray[np.float32], keys: Sequence[ProductKey]
    ) -> list[tuple[ProductKey, float]]:
        if scores.shape != (len(self.keys),):
            raise ValueError("Score/product alignment mismatch")
        if len(set(keys)) != len(keys) or any(key not in self.row_for_key for key in keys):
            raise ValueError("Unknown or duplicate candidate")
        return [(key, float(scores[self.row_for_key[key]])) for key in keys]

    def probe_hashes(self) -> dict[str, str]:
        return {
            query: hashlib.sha256(self.score_all(query).tobytes()).hexdigest()
            for query in PROBE_QUERIES
        }


def build(products: pl.DataFrame, output: Path, origin: Mapping[str, str]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Index destination exists; preserve prior bundles")
    expected = ["product_locale", "product_id", "product_title"]
    if any(name not in products.columns or products[name].dtype != pl.String for name in expected):
        raise ValueError("Invalid catalog schema")
    products = products.select(expected).sort("product_locale", "product_id")
    if not products.height or products.null_count().row(0) != (0, 0, 0):
        raise ValueError("Catalog must be nonempty with normalized missing text")
    keys = products.select("product_locale", "product_id")
    if keys.unique().height != products.height or set(products["product_locale"]) != {"us"}:
        raise ValueError("Expected unique US product identities")
    if products["product_id"].str.strip_chars().eq("").any():
        raise ValueError("Empty product identity")
    started = time.perf_counter()
    vocab: dict[str, int] = {}
    documents: list[list[int]] = []
    tokens_total = 0
    empty_documents = 0
    for i, title in enumerate(products["product_title"]):
        ids = [vocab.setdefault(token, len(vocab)) for token in analyze(title)]
        documents.append(ids)
        tokens_total += len(ids)
        empty_documents += not ids
        if i % 10000 == 0:
            check_memory()
    if not vocab:
        raise ValueError("Catalog contains no indexable title tokens")
    backend = bm25s.BM25(
        k1=1.5,
        b=0.75,
        method="lucene",
        idf_method="lucene",
        dtype="float32",
        int_dtype="int32",
        backend="numpy",
    )
    backend.index((documents, vocab), create_empty_token=False, show_progress=False)
    del documents
    gc.collect()
    check_memory()
    output.mkdir(parents=True)
    backend.save(output / "backend", allow_pickle=False)
    products.write_parquet(output / "products.parquet", compression="zstd", row_group_size=8192)
    metadata = seal(
        output,
        {
            "schema_version": 1,
            "kind": "atlasrank.bm25",
            **SCORING,
            "bm25s_version": importlib.metadata.version("bm25s"),
            "numpy_version": np.__version__,
            "scipy_version": importlib.metadata.version("scipy"),
            "num_docs": products.height,
            "origin": dict(origin),
        },
    )
    index = LexicalIndex(backend, products, metadata)
    return {
        "schema_version": 1,
        "bundle_id": index.bundle_id,
        "num_docs": products.height,
        "vocabulary_terms": len(vocab),
        "title_tokens": tokens_total,
        "empty_title_documents": empty_documents,
        "nonzero_term_document_scores": len(backend.scores["data"]),
        "elapsed_seconds": time.perf_counter() - started,
        "probe_hashes_before_load": index.probe_hashes(),
    }


def load(directory: Path) -> LexicalIndex:
    manifest = validate(directory)
    try:
        products = pl.read_parquet(directory / "products.parquet")
        if (
            products.columns != ["product_locale", "product_id", "product_title"]
            or any(dtype != pl.String for dtype in products.dtypes)
            or products.height != manifest["num_docs"]
            or products.null_count().row(0) != (0, 0, 0)
        ):
            raise InvalidIndex("Catalog schema/count mismatch")
        params = json.loads((directory / "backend/params.index.json").read_text())
        if any(params.get(k) != manifest[k] for k in ("method", "k1", "b", "dtype", "backend")):
            raise InvalidIndex("Stored BM25 parameters mismatch")
        if params.get("idf_method") != "lucene" or params.get("num_docs") != products.height:
            raise InvalidIndex("Stored IDF configuration mismatch")
        if params.get("version") != manifest["bm25s_version"] or params.get("int_dtype") != "int32":
            raise InvalidIndex("Stored library/index dtype configuration mismatch")
        backend = bm25s.BM25.load(directory / "backend", mmap=True, allow_pickle=False)
        scores = backend.scores
        n = products.height
        if (
            scores["num_docs"] != n
            or any(scores[name].ndim != 1 for name in ("data", "indices", "indptr"))
            or scores["data"].dtype != np.float32
            or scores["indices"].dtype != np.int32
            or scores["indptr"].dtype != np.int32
            or len(scores["data"]) != len(scores["indices"])
            or len(scores["indptr"]) != len(backend.vocab_dict) + 1
            or scores["indptr"][0] != 0
            or scores["indptr"][-1] != len(scores["data"])
            or np.any(np.diff(scores["indptr"]) < 0)
            or np.any(scores["indices"] < 0)
            or np.any(scores["indices"] >= n)
            or not np.isfinite(scores["data"]).all()
            or np.any(scores["data"] < 0)
        ):
            raise InvalidIndex("Sparse score structure mismatch")
        if set(backend.vocab_dict.values()) != set(range(len(backend.vocab_dict))) or any(
            not isinstance(term, str) or not term for term in backend.vocab_dict
        ):
            raise InvalidIndex("Stored vocabulary identity mismatch")
        return LexicalIndex(backend, products, manifest)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        pl.exceptions.PolarsError,
    ) as error:
        raise InvalidIndex("Failed to load a compatible lexical bundle") from error
