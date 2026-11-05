"""Exact fixture-scale FAISS adapter with explicit cutoff ties and product alignment."""

from collections.abc import Sequence
from typing import Any, Protocol, cast

import faiss
import numpy as np
from numpy.typing import NDArray

from atlasrank.evaluation.metrics import ProductKey
from atlasrank.retrieval.encoder import validate_vectors


class _SubsetScorer(Protocol):
    # Present in the pinned native API but omitted from the wheel's type stub.
    def compute_distance_subset(
        self, n: int, x: Any, k: int, distances: Any, labels: Any
    ) -> None: ...


class DenseIndex:
    def __init__(self, keys: Sequence[ProductKey], vectors: NDArray[np.float32]) -> None:
        if not keys or len(keys) != len(vectors) or list(keys) != sorted(set(keys)):
            raise ValueError("Expected unique canonical product keys aligned to vector rows")
        if any(locale != "us" or not pid.strip() for locale, pid in keys):
            raise ValueError("Expected nonempty US identities")
        if vectors.ndim != 2 or vectors.shape[1] < 1:
            raise ValueError("Expected a nonempty vector dimension")
        validate_vectors(vectors, vectors.shape[1])
        faiss.omp_set_num_threads(1)
        self.keys = tuple(keys)
        self.row_for_key = {key: row for row, key in enumerate(self.keys)}
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors))

    @classmethod
    def from_backend(cls, keys: Sequence[ProductKey], backend: faiss.IndexFlatIP) -> "DenseIndex":
        if (
            not keys
            or list(keys) != sorted(set(keys))
            or any(locale != "us" or not pid.strip() for locale, pid in keys)
        ):
            raise ValueError("Expected canonical US product keys")
        if (
            not isinstance(backend, faiss.IndexFlatIP)
            or backend.metric_type != faiss.METRIC_INNER_PRODUCT
            or backend.ntotal != len(keys)
            or backend.d < 1
        ):
            raise ValueError("Incompatible exact dense index")
        result = cls.__new__(cls)
        result.keys = tuple(keys)
        result.row_for_key = {key: row for row, key in enumerate(keys)}
        result.index = backend
        faiss.omp_set_num_threads(1)
        return result

    def candidate_scores(
        self, vector: NDArray[np.float32], keys: Sequence[ProductKey]
    ) -> list[tuple[ProductKey, float]]:
        validate_vectors(vector, self.index.d)
        if (
            vector.shape[0] != 1
            or len(set(keys)) != len(keys)
            or any(key not in self.row_for_key for key in keys)
        ):
            raise ValueError("Invalid query/candidate alignment")
        if not keys:
            return []
        rows = np.array([[self.row_for_key[key] for key in keys]], dtype=np.int64)
        scores = np.empty((1, len(keys)), dtype=np.float32)
        query = np.ascontiguousarray(vector)
        cast(_SubsetScorer, self.index).compute_distance_subset(
            1, faiss.swig_ptr(query), len(keys), faiss.swig_ptr(scores), faiss.swig_ptr(rows)
        )
        return [(key, float(score)) for key, score in zip(keys, scores[0], strict=True)]

    def search(self, vector: NDArray[np.float32], k: int = 100) -> list[tuple[ProductKey, float]]:
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
            raise ValueError("top_k must be an integer from 1 to 100")
        validate_vectors(vector, self.index.d)
        if vector.shape[0] != 1:
            raise ValueError("Reference query batch size is 1")
        count = min(k, len(self.keys))
        scores, rows = self.index.search(np.ascontiguousarray(vector), count)
        threshold = np.float32(scores[0, -1])
        radius = float(np.nextafter(threshold, np.float32(-np.inf)))
        _, range_scores, range_rows = self.index.range_search(vector, radius)
        candidates = {
            int(row): float(score) for row, score in zip(range_rows, range_scores, strict=True)
        }
        for row, score in zip(rows[0], scores[0], strict=True):
            if candidates.get(int(row)) != float(score):
                raise RuntimeError("FAISS search/range score consistency failed")
        ordered = sorted(candidates, key=lambda row: (-candidates[row], self.keys[row]))[:count]
        return [(self.keys[row], candidates[row]) for row in ordered]
