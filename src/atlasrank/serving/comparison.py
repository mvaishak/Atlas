"""Small, pinned development replays. No inference, scoring or relevance computation."""

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlasrank.artifacts.dense import digest, file_hash

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REPO = "sentence-transformers/all-MiniLM-L6-v2"
MAX_FIXTURE_BYTES = 1024**2
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class Product(StrictModel):
    rank: int = Field(ge=1, le=10)
    product_locale: Literal["us"]
    product_id: str = Field(min_length=1)
    title: str
    score: float = Field(gt=0)
    judgment: Literal["E", "S", "C", "I", "unknown"]


class Pipeline(StrictModel):
    pipeline_id: str
    bundle_id: str
    score_kind: Literal["bm25", "rrf60"]
    historical_search_ms: None = None


class Example(StrictModel):
    query_id: int = Field(ge=0)
    query: str = Field(min_length=1)
    group_sha256: Hash
    rankings: dict[Literal["bm25", "hybrid"], list[Product]]

    @model_validator(mode="after")
    def valid_rankings(self) -> Self:
        if set(self.rankings) != {"bm25", "hybrid"}:
            raise ValueError("Both frozen pipelines required")
        for hits in self.rankings.values():
            keys = [(hit.product_locale, hit.product_id) for hit in hits]
            if len(hits) > 10 or len(set(keys)) != len(keys):
                raise ValueError("Unique top-10 identities required")
            if [hit.rank for hit in hits] != list(range(1, len(hits) + 1)):
                raise ValueError("Original rank positions required")
            if any(a.score < b.score for a, b in zip(hits, hits[1:], strict=False)):
                raise ValueError("Scores must retain descending retrieval order")
        return self


class Sources(StrictModel):
    report_sha256: Hash
    per_query_sha256: Hash
    title_projection_sha256: Hash
    development_sha256: Hash
    evaluation_contract_sha256: Hash
    provenance: dict[str, Any]


class Comparison(StrictModel):
    schema_version: Literal[1]
    partition: Literal["development"]
    track: Literal["B"]
    mode: Literal["saved_development"]
    evaluation_contract: Literal["v0.3"]
    retriever_k: Literal[100]
    rrf_constant: Literal[60]
    sources: Sources
    pipelines: dict[Literal["bm25", "hybrid"], Pipeline]
    examples: list[Example] = Field(min_length=5, max_length=5)
    fixture_id: Hash

    @model_validator(mode="after")
    def compatible(self) -> Self:
        if len({e.query_id for e in self.examples}) != 5:
            raise ValueError("Exactly five distinct development queries required")
        p = self.sources.provenance
        for name in ["product_recipe", "query_recipe"]:
            if p[name]["revision"] != REVISION or p[name]["repo"] != REPO:
                raise ValueError("Frozen M4 pretrained provenance required")
        if (
            p["product_recipe_id"] != digest(p["product_recipe"])
            or p["recipe_id"] != digest(p["query_recipe"])
            or p["source"]["development_sha256"] != self.sources.development_sha256
        ):
            raise ValueError("Recipe or development provenance differs")
        lexical, dense = p["lexical_artifact_id"], p["dense_artifact_id"]
        expected: dict[Literal["bm25", "hybrid"], tuple[str, str, str]] = {
            "bm25": ("bm25-title-lucene-v1", lexical, "bm25"),
            "hybrid": ("rrf60-m4-pretrained-v1", f"rrf60:{lexical}:{dense}", "rrf60"),
        }
        if set(self.pipelines) != set(expected):
            raise ValueError("Frozen pipeline pair required")
        for name, (pipeline, bundle, score) in expected.items():
            item = self.pipelines[name]
            if (item.pipeline_id, item.bundle_id, item.score_kind) != (pipeline, bundle, score):
                raise ValueError("Pipeline identity differs")
        payload = self.model_dump(exclude={"fixture_id"})
        if digest(payload) != self.fixture_id:
            raise ValueError("Fixture integrity differs")
        return self


def load_comparison(path: Path, expected_sha256: str | None, lexical_bundle_id: str) -> Comparison:
    if (
        path.is_symlink()
        or path.stat().st_size > MAX_FIXTURE_BYTES
        or expected_sha256 is None
        or file_hash(path) != expected_sha256
    ):
        raise ValueError("Missing, oversized or unpinned comparison fixture")
    fixture = Comparison.model_validate(json.loads(path.read_text()))
    if fixture.pipelines["bm25"].bundle_id != lexical_bundle_id:
        raise ValueError("Comparison differs from loaded lexical bundle")
    return fixture
