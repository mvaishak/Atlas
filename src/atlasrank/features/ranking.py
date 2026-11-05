"""Five approved features, canonical identities and fail-closed source binding."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from atlasrank.artifacts.dense import digest
from atlasrank.data.splits import normalize_query
from atlasrank.evaluation.metrics import ProductKey
from atlasrank.retrieval.analyzer import ANALYZER, analyze
from atlasrank.retrieval.bm25 import LexicalIndex
from atlasrank.retrieval.chunks import Encoder
from atlasrank.retrieval.encoder import validate_vectors

FEATURE_NAMES = (
    "bm25_score",
    "dense_cosine",
    "title_query_token_overlap",
    "query_token_count",
    "title_token_count",
)
SCHEMA: dict[str, Any] = {
    "version": 1,
    "names": list(FEATURE_NAMES),
    "dtype": "float32",
    "analyzer": ANALYZER,
    "row_order": "product_locale_product_id",
    "overlap": "unique_token_jaccard_empty_union_zero",
    "lengths": "analyzer_tokens_with_repetition",
    "numeric_missing": "forbidden",
    "empty_text": "normalized_empty_string_distinct_from_no_analyzer_tokens",
    "dense": "native_indexflatip_subset_no_clipping",
}
SCHEMA_ID = digest(SCHEMA)


class DenseScorer(Protocol):
    keys: tuple[ProductKey, ...]

    def candidate_scores(
        self, vector: NDArray[np.float32], keys: Sequence[ProductKey]
    ) -> list[tuple[ProductKey, float]]: ...


def model_identity(recipe: dict[str, Any]) -> str:
    files = recipe.get("model_files_sha256", recipe.get("model_files", {}))
    revision = recipe.get("revision", recipe.get("base_revision"))
    if not revision or not files.get("model.safetensors") or recipe.get("dimensions", 0) < 1:
        raise ValueError("Model weights, revision and dimensions must be pinned")
    return digest(
        {
            "revision": revision,
            "model_files_sha256": files,
            "dimensions": recipe["dimensions"],
        }
    )


@dataclass(frozen=True)
class DenseBinding:
    """Trusted metadata from a validated dense bundle, not caller query data.

    For actual bundles the existing bundle loader must validate native vectors/IDs
    and source titles before this binding is constructed. This phase uses fixtures.
    """

    artifact_id: str
    catalog_sha256: str
    catalog_content_id: str
    product_recipe_id: str
    model_id: str


@dataclass(frozen=True)
class Provenance:
    schema_id: str
    catalog_sha256: str
    catalog_content_id: str
    lexical_bundle_id: str
    dense_bundle_id: str
    query_recipe_id: str
    product_recipe_id: str
    model_id: str
    model_revision: str
    model_weights_sha256: str
    dimensions: int


class FeatureContext:
    def __init__(
        self, lexical: LexicalIndex, dense: DenseScorer, encoder: Encoder, binding: DenseBinding
    ) -> None:
        self.lexical, self.dense, self.encoder, self.binding = lexical, dense, encoder, binding
        recipe = encoder.recipe
        files = recipe.get("model_files_sha256", recipe.get("model_files", {}))
        revision = recipe.get("revision", recipe.get("base_revision", ""))
        self.provenance = Provenance(
            SCHEMA_ID,
            binding.catalog_sha256,
            binding.catalog_content_id,
            lexical.bundle_id,
            binding.artifact_id,
            digest(recipe),
            binding.product_recipe_id,
            binding.model_id,
            revision,
            files.get("model.safetensors", ""),
            recipe.get("dimensions", 0),
        )
        self.validate()

    def validate(self) -> None:
        content = digest(list(zip(self.lexical.keys, self.lexical.titles, strict=True)))
        if (
            tuple(self.lexical.keys) != self.dense.keys
            or content != self.binding.catalog_content_id
            or self.lexical.metadata["origin"]["catalog_sha256"] != self.binding.catalog_sha256
            or model_identity(self.encoder.recipe) != self.binding.model_id
            or digest(self.encoder.recipe) != self.provenance.query_recipe_id
            or self.lexical.bundle_id != self.provenance.lexical_bundle_id
        ):
            raise ValueError("Incompatible source/catalog/encoder binding")
        if any(not value for value in self.provenance.__dict__.values()):
            raise ValueError("Complete source/model provenance required")


@dataclass(frozen=True)
class FeatureBatch:
    names: tuple[str, ...]
    keys: tuple[ProductKey, ...]
    candidate_order: tuple[ProductKey, ...]
    values: NDArray[np.float32]
    missing: NDArray[np.bool_]
    text_empty: NDArray[np.bool_]
    query_fingerprint: str
    provenance: Provenance
    schema_id: str = SCHEMA_ID

    def __post_init__(self) -> None:
        n = len(self.keys)
        if (
            self.schema_id != SCHEMA_ID
            or self.names != FEATURE_NAMES
            or n == 0
            or list(self.keys) != sorted(set(self.keys))
            or len(self.candidate_order) != n
            or set(self.candidate_order) != set(self.keys)
            or any(loc != "us" or not pid.strip() for loc, pid in self.keys)
            or self.provenance.schema_id != SCHEMA_ID
        ):
            raise ValueError("Feature schema or identity alignment differs")
        if (
            self.values.dtype != np.float32
            or self.values.shape != (n, 5)
            or not self.values.flags.c_contiguous
            or not np.isfinite(self.values).all()
            or self.missing.dtype != np.bool_
            or self.missing.shape != (n, 5)
            or self.missing.any()
            or self.text_empty.dtype != np.bool_
            or self.text_empty.shape != (n, 2)
            or not self.query_fingerprint
        ):
            raise ValueError("Invalid feature dtype, values or missing semantics")
        if (
            (self.values[:, 0] < 0).any()
            or (self.values[:, 2:] < 0).any()
            or (self.values[:, 2] > 1).any()
            or not np.array_equal(self.values[:, 3:], np.floor(self.values[:, 3:]))
        ):
            raise ValueError("Invalid overlap/count/lexical values")
        for name in ["values", "missing", "text_empty"]:
            value = getattr(self, name).copy(order="C")
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def input_rows(self) -> tuple[int, ...]:
        rows = {key: i for i, key in enumerate(self.keys)}
        return tuple(rows[key] for key in self.candidate_order)


def required_scores(
    pairs: list[tuple[ProductKey, float]], keys: tuple[ProductKey, ...]
) -> dict[ProductKey, float]:
    if (
        len(pairs) != len(keys)
        or len({key for key, _ in pairs}) != len(keys)
        or {key for key, _ in pairs} != set(keys)
        or any(score is None or not np.isfinite(score) for _, score in pairs)
    ):
        raise ValueError("Missing, duplicate or non-finite required score")
    return dict(pairs)


def build_features(
    context: FeatureContext, query: str, candidates: Sequence[ProductKey]
) -> FeatureBatch:
    # Full catalog/title alignment is checked at context construction and readiness.
    # Avoid re-hashing the full catalog for each bounded candidate request.
    if (
        digest(context.encoder.recipe) != context.provenance.query_recipe_id
        or model_identity(context.encoder.recipe) != context.provenance.model_id
        or context.lexical.bundle_id != context.provenance.lexical_bundle_id
    ):
        raise ValueError("Encoder/source provenance changed after readiness")
    ordered = tuple(candidates)
    if (
        not ordered
        or len(set(ordered)) != len(ordered)
        or not set(ordered).issubset(context.lexical.catalog)
    ):
        raise ValueError("Unknown or duplicate candidate identity")
    keys = tuple(sorted(ordered))
    normalized = normalize_query(query)
    lexical = required_scores(
        context.lexical.candidate_scores(context.lexical.score_all(query), keys), keys
    )
    vector = context.encoder.encode([query])
    validate_vectors(vector, int(context.encoder.recipe["dimensions"]))
    if vector.shape[0] != 1:
        raise ValueError("Exactly one query vector required")
    dense = required_scores(context.dense.candidate_scores(vector, keys), keys)
    qt = analyze(query)
    qs = set(qt)
    values, empty = [], []
    for key in keys:
        title = context.lexical.titles[context.lexical.row_for_key[key]]
        tt = analyze(title)
        union = qs | set(tt)
        overlap = len(qs & set(tt)) / len(union) if union else 0.0
        values.append([lexical[key], dense[key], overlap, len(qt), len(tt)])
        empty.append([not normalized, not normalize_query(title)])
    return FeatureBatch(
        FEATURE_NAMES,
        keys,
        ordered,
        np.array(values, dtype=np.float32),
        np.zeros((len(keys), 5), dtype=np.bool_),
        np.array(empty, dtype=np.bool_),
        digest(normalized),
        context.provenance,
    )
