"""Independent bounded resume, integrity and development-track fixtures; no downloads."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from atlasrank.artifacts.dense import InvalidDense, digest, file_hash, seal, write_once
from atlasrank.retrieval.bm25 import build as build_lexical
from atlasrank.retrieval.bm25 import load as load_lexical
from atlasrank.retrieval.chunks import (
    accounted_seconds,
    canonical_catalog,
    completed_chunks,
    encode_chunks,
)
from atlasrank.retrieval.comparison import (
    compare_development,
    evaluate_development,
    verify_bm25_reference,
)
from atlasrank.retrieval.dense_bundle import BUNDLE_FILES, build_bundle, load_bundle


class FixtureEncoder:
    def __init__(self, fail_call: int | None = None) -> None:
        self.recipe: dict[str, Any] = {
            "kind": "synthetic_fixture",
            "dimensions": 3,
            "batch_size": 2,
            "device": "cpu",
            "preprocessing": "fixture_v1",
        }
        self.calls = 0
        self.fail_call = fail_call

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        self.calls += 1
        if self.calls == self.fail_call:
            raise InterruptedError("Injected interruption during an unpublished chunk")
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vector = np.array([1 + h[0], 1 + h[1], 1 + h[2]], dtype=np.float32)
            result.append(vector / np.linalg.norm(vector))
        return np.asarray(result, dtype=np.float32)


def catalog(path: Path, count: int = 8) -> Path:
    pl.DataFrame(
        {
            "product_locale": ["us"] * count,
            "product_id": [f"{i:04d}" for i in range(count)],
            "product_title": [f"apple {i}" for i in range(count)],
        }
    ).write_parquet(path)
    return path


def build_fixture(tmp_path: Path) -> tuple[Path, Path, FixtureEncoder]:
    source = catalog(tmp_path / "catalog.parquet")
    encoder = FixtureEncoder()
    job = tmp_path / "job"
    encode_chunks(
        source, job, encoder, {"scope": "synthetic_fixture"}, chunk_rows=4, wall_seconds=60
    )
    bundle = tmp_path / "bundle"
    build_bundle(job, bundle)
    return job, bundle, encoder


def test_interrupted_mid_chunk_resume_exact_and_no_reencoding(tmp_path: Path) -> None:
    source = catalog(tmp_path / "catalog.parquet")
    interrupted = FixtureEncoder(fail_call=4)
    resumed = tmp_path / "resumed"
    with pytest.raises(InterruptedError):
        encode_chunks(
            source,
            resumed,
            interrupted,
            {"scope": "synthetic_fixture"},
            chunk_rows=4,
            wall_seconds=60,
        )
    assert len(list((resumed / "chunks").iterdir())) == 1
    abandoned = list((resumed / "partials").iterdir())
    assert len(abandoned) == 1 and not (abandoned[0] / "COMPLETE.json").exists()
    committed = next((resumed / "chunks").iterdir())
    before = {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in committed.iterdir()}
    assert all(p.stat().st_mode & 0o222 == 0 for p in committed.iterdir())
    fresh_encoder = FixtureEncoder()
    result = encode_chunks(
        source,
        resumed,
        fresh_encoder,
        {"scope": "synthetic_fixture"},
        chunk_rows=4,
        wall_seconds=60,
    )
    assert fresh_encoder.calls == 2 and result["new_chunks"] == 1
    assert before == {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in committed.iterdir()}
    assert abandoned[0].exists()
    uninterrupted = tmp_path / "uninterrupted"
    other = encode_chunks(
        source,
        uninterrupted,
        FixtureEncoder(),
        {"scope": "synthetic_fixture"},
        chunk_rows=4,
        wall_seconds=60,
    )
    assert result["job_id"] == other["job_id"] and result["chunks"] == other["chunks"]
    for a, b in zip(completed_chunks(resumed), completed_chunks(uninterrupted), strict=True):
        assert a[0] == b[0] and a[1].equals(b[1])
        np.testing.assert_array_equal(a[2], b[2])
    resumed_bundle, uninterrupted_bundle = tmp_path / "bundle-r", tmp_path / "bundle-u"
    assert (
        build_bundle(resumed, resumed_bundle)["artifact_id"]
        == build_bundle(uninterrupted, uninterrupted_bundle)["artifact_id"]
    )
    assert accounted_seconds(resumed) > 0
    # Successful resume of a complete job performs no encoder calls.
    again = FixtureEncoder()
    assert (
        encode_chunks(
            source, resumed, again, {"scope": "synthetic_fixture"}, chunk_rows=4, wall_seconds=60
        )["new_chunks"]
        == 0
    )
    assert again.calls == 0


@pytest.mark.parametrize("change", ["source", "recipe", "origin", "chunk_rows", "budget"])
def test_resume_rejects_changed_contract(tmp_path: Path, change: str) -> None:
    source = catalog(tmp_path / "catalog.parquet")
    encoder = FixtureEncoder()
    job = tmp_path / "job"
    encode_chunks(source, job, encoder, {"scope": "fixture"}, chunk_rows=4, wall_seconds=60)
    origin = {"scope": "fixture"}
    chunk_rows, budget = 4, 60
    if change == "source":
        pl.read_parquet(source).with_columns(
            pl.lit("changed").alias("product_title")
        ).write_parquet(source)
    elif change == "recipe":
        encoder.recipe = {**encoder.recipe, "device": "other"}
    elif change == "origin":
        origin = {"scope": "other"}
    elif change == "chunk_rows":
        chunk_rows = 2
    else:
        budget = 61
    with pytest.raises(InvalidDense):
        encode_chunks(source, job, encoder, origin, chunk_rows=chunk_rows, wall_seconds=budget)


def test_partial_files_preserved_and_corrupt_committed_chunk_fails(tmp_path: Path) -> None:
    job, _, encoder = build_fixture(tmp_path)
    chunk = next((job / "chunks").iterdir())
    p = chunk / "vectors.npy"
    p.chmod(0o644)
    with p.open("ab") as target:
        target.write(b"changed")
    with pytest.raises(InvalidDense):
        encode_chunks(
            tmp_path / "catalog.parquet",
            job,
            encoder,
            {"scope": "synthetic_fixture"},
            chunk_rows=4,
            wall_seconds=60,
        )


def test_canonical_snapshot_and_cumulative_crash_accounting(tmp_path: Path) -> None:
    source = catalog(tmp_path / "source.parquet")
    frame = pl.read_parquet(source)
    frame.reverse().write_parquet(source)
    canonical = canonical_catalog(source, tmp_path / "canonical.parquet")
    assert pl.read_parquet(canonical).equals(frame)
    job = tmp_path / "job"
    encode_chunks(canonical, job, FixtureEncoder(), {}, chunk_rows=4, wall_seconds=60)
    write_once(job / "attempts" / "crash.start.json", {"utc_start": 0, "reserved_seconds": 60})
    with pytest.raises(RuntimeError, match="budget"):
        encode_chunks(canonical, job, FixtureEncoder(), {}, chunk_rows=4, wall_seconds=60)
    assert (job / "attempts" / "crash.recovered.json").exists()
    assert accounted_seconds(job) >= 60


def test_writer_lock_and_mid_call_recipe_change_fail_closed(tmp_path: Path) -> None:
    import fcntl

    source = catalog(tmp_path / "source.parquet")
    locked = tmp_path / "locked"
    locked.mkdir()
    with (locked / ".writer.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active writer"):
            encode_chunks(source, locked, FixtureEncoder(), {}, chunk_rows=4)
    assert not (locked / "JOB.json").exists()

    class ChangingEncoder(FixtureEncoder):
        def encode(self, texts: list[str]) -> NDArray[np.float32]:
            values = super().encode(texts)
            self.recipe["device"] = "changed"
            return values

    destination = tmp_path / "changed"
    with pytest.raises(InvalidDense, match="changed during"):
        encode_chunks(source, destination, ChangingEncoder(), {}, chunk_rows=4)
    assert not list((destination / "chunks").iterdir())
    assert len(list((destination / "partials").iterdir())) == 1


def test_full_run_guard_precedes_any_model_or_data_work(tmp_path: Path) -> None:
    from atlasrank.retrieval.full_run import approved_manifest, supervise

    manifest = tmp_path / "not-approved.json"
    manifest.write_text(json.dumps({"execution_approved": False}))
    with pytest.raises(PermissionError, match="explicit lead"):
        supervise(manifest)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["not-approved.json"]
    wrong = {
        "execution_approved": True,
        "schema_version": 1,
        "pipeline": "m4-cpu-reference-v1",
        "data_root": "data/esci",
        "wall_seconds": 43200,
        "rss_limit_bytes": 8 * 1024**3,
        "additional_artifact_limit_bytes": 8 * 1024**3,
        "available_ram_start_bytes": 6 * 1024**3,
        "available_ram_stop_bytes": 1024**3,
        "implementation_sha256": {},
    }
    manifest.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="implementation/lock"):
        approved_manifest(manifest)


def test_worker_guard_rejects_orphan_before_resource_or_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlasrank.retrieval.full_run import worker_guard

    monkeypatch.delenv("ATLAS_M4_SUPERVISOR_PID", raising=False)
    monkeypatch.delenv("ATLAS_M4_DEADLINE_MONOTONIC", raising=False)
    with pytest.raises(RuntimeError, match="supervisor"):
        worker_guard({})()


def test_missing_chunk_and_rehashed_chunk_mapping_mismatch_fail(tmp_path: Path) -> None:
    job, _, encoder = build_fixture(tmp_path)
    chunk = job / "chunks/000000000004"
    chunk.rename(job / "partials/preserved-removed-chunk")
    with pytest.raises((InvalidDense, FileNotFoundError)):
        list(completed_chunks(job))
    preserved = job / "partials/preserved-removed-chunk"
    preserved.rename(chunk)
    # Rehashing a changed title doesn't defeat comparison to expected source rows on resume.
    metadata = json.loads((chunk / "manifest.json").read_text())
    metadata = {k: v for k, v in metadata.items() if k not in {"files", "artifact_id"}}
    (chunk / "products.parquet").chmod(0o644)
    pl.read_parquet(chunk / "products.parquet").with_columns(
        pl.lit("wrong").alias("product_title")
    ).write_parquet(chunk / "products.parquet")
    (chunk / "manifest.json").rename(chunk / "old-manifest")
    (chunk / "COMPLETE.json").rename(chunk / "old-complete")
    # Preserve old metadata outside the component directory before resealing this synthetic fixture.
    (chunk / "old-manifest").rename(job / "old-manifest")
    (chunk / "old-complete").rename(job / "old-complete")
    seal(chunk, metadata, {"products.parquet", "vectors.npy"})
    with pytest.raises(InvalidDense):
        encode_chunks(
            tmp_path / "catalog.parquet",
            job,
            encoder,
            {"scope": "synthetic_fixture"},
            chunk_rows=4,
            wall_seconds=60,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "corrupt",
        "extra",
        "recipe",
        "version",
        "index_type",
        "row_alignment",
        "norm",
        "identities",
        "symlink",
    ],
)
def test_dense_bundle_fails_closed(tmp_path: Path, damage: str) -> None:
    _, bundle, encoder = build_fixture(tmp_path)
    original = json.loads((bundle / "manifest.json").read_text())
    metadata = {k: v for k, v in original.items() if k not in {"files", "artifact_id"}}
    if damage == "missing":
        (bundle / "COMPLETE.json").rename(bundle / "missing")
    elif damage == "corrupt":
        (bundle / "index.faiss").chmod(0o644)
        with (bundle / "index.faiss").open("ab") as target:
            target.write(b"corrupt")
    elif damage == "extra":
        (bundle / "extra.txt").write_text("extra")
    elif damage == "symlink":
        (bundle / "vectors.npy").rename(tmp_path / "vectors.npy")
        os.symlink(tmp_path / "vectors.npy", bundle / "vectors.npy")
    else:
        mutable = tmp_path / "mutated"
        mutable.mkdir()
        for name in BUNDLE_FILES:
            shutil.copyfile(bundle / name, mutable / name)
        bundle = mutable
        if damage == "recipe":
            metadata["recipe_id"] = "wrong"
        elif damage == "version":
            metadata["faiss_version"] = "wrong"
        elif damage == "index_type":
            faiss.write_index(faiss.IndexFlatL2(3), str(bundle / "index.faiss"))
        elif damage == "row_alignment":
            vectors = np.load(bundle / "vectors.npy", allow_pickle=False)
            index = faiss.IndexFlatIP(3)
            index.add(vectors[::-1].copy())
            faiss.write_index(index, str(bundle / "index.faiss"))
        elif damage == "norm":
            vectors = np.load(bundle / "vectors.npy", allow_pickle=False)
            vectors[0] = 0
            np.save(bundle / "vectors.npy", vectors, allow_pickle=False)
        else:
            products = pl.read_parquet(bundle / "products.parquet")
            products.with_columns(pl.lit("duplicate").alias("product_id")).write_parquet(
                bundle / "products.parquet"
            )
        seal(bundle, metadata, BUNDLE_FILES)
    with pytest.raises(InvalidDense):
        load_bundle(bundle, encoder.recipe)


def test_bundle_save_load_native_candidates_and_expected_origin(tmp_path: Path) -> None:
    _, bundle, encoder = build_fixture(tmp_path)
    index, products, manifest = load_bundle(
        bundle, encoder.recipe, expected_origin={"scope": "synthetic_fixture"}
    )
    query = encoder.encode([products["product_title"][0]])
    hits = index.search(query, 8)
    assert index.candidate_scores(query, [key for key, _ in hits]) == hits
    assert load_bundle(bundle, encoder.recipe)[0].search(query, 8) == hits
    assert manifest["rows"] == 8
    with pytest.raises(InvalidDense):
        load_bundle(bundle, encoder.recipe, expected_origin={"scope": "other"})
    with pytest.raises(ValueError):
        index.candidate_scores(query, [("us", "unknown")])


@pytest.mark.parametrize("all_irrelevant", [False, True])
def test_development_job_tracks_all_candidates_unknowns_and_undefined_recall(
    tmp_path: Path, all_irrelevant: bool
) -> None:
    # More than 100 judged products proves A never uses B's truncation.
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    source = catalog(prepared / "catalog.parquet", 120)
    products = pl.read_parquet(source)
    development = pl.DataFrame(
        {
            "query_id": [1] * 110 + [2],
            "query": ["apple"] * 110 + ["oovzz"],
            "product_locale": ["us"] * 111,
            "product_id": [f"{i:04d}" for i in range(10, 120)] + ["0000"],
            "esci_label": ["I" if all_irrelevant else "E"] * 110 + ["I"],
            "group_sha256": ["g1"] * 110 + ["g2"],
            "partition": ["development"] * 111,
        }
    )
    development.write_parquet(prepared / "development.parquet")
    write_once(
        prepared / "audit.json",
        {
            "schema_version": 1,
            "evaluation_contract": "v0.3",
            "output_sha256": {"development.parquet": file_hash(prepared / "development.parquet")},
        },
    )
    write_once(
        prepared / "COMPLETE.json",
        {"schema_version": 1, "audit_sha256": file_hash(prepared / "audit.json")},
    )
    pins = {
        "scope": "synthetic_fixture",
        "catalog_sha256": file_hash(source),
        "prepared_audit_sha256": file_hash(prepared / "audit.json"),
        "development_sha256": file_hash(prepared / "development.parquet"),
    }
    encoder = FixtureEncoder()
    job = tmp_path / "job"
    origin = {k: pins[k] for k in ("catalog_sha256", "prepared_audit_sha256")}
    encode_chunks(source, job, encoder, origin, chunk_rows=4, wall_seconds=60)
    bundle = tmp_path / "dense"
    build_bundle(job, bundle)
    lexical = tmp_path / "lexical"
    build_lexical(products, lexical, origin)
    report = evaluate_development(
        prepared, lexical, bundle, encoder, tmp_path / "report", pins, job
    )
    rows = [json.loads(line) for line in (tmp_path / "report/per-query.jsonl").open()]
    reference = tmp_path / "m3-fixture-reference.jsonl"
    reference.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": r["query_id"],
                    "group_sha256": r["group_sha256"],
                    "A": r["metrics"]["A"]["bm25"],
                    "B": r["metrics"]["B"]["bm25"],
                    "returned_top100": [
                        {"product_locale": key[0], "product_id": key[1], "score": score}
                        for key, score in r["B_scores"]["bm25"]
                    ],
                }
            )
            + "\n"
            for r in rows
        )
    )
    verify_bm25_reference(tmp_path / "report", reference)
    corrupted = [json.loads(line) for line in reference.open()]
    corrupted[0]["returned_top100"][0]["score"] += 1
    reference.write_text("".join(json.dumps(r) + "\n" for r in corrupted))
    with pytest.raises(InvalidDense, match="frozen M3"):
        verify_bm25_reference(tmp_path / "report", reference)
    assert len(rows[0]["A_rankings"]["hybrid"]) == 110
    assert len(rows[0]["B_rankings"]["hybrid"]) == 100
    assert rows[0]["metrics"]["B"]["bm25"]["judgment_coverage_at_100"] == 0.9
    assert "ndcg_at_10" not in report["B"]["hybrid"]
    assert rows[1]["B_rankings"]["bm25"] == []
    assert report["A"]["dense"]["queries"] == 2
    assert report["A"]["dense"]["zero_idcg_queries"] == (2 if all_irrelevant else 1)
    assert report["B"]["dense"]["recall_excluded_queries"] == (2 if all_irrelevant else 1)
    if all_irrelevant:
        assert (
            report["paired_intervals"]["B.known_exact_recall_at_100.dense_minus_bm25"]["delta"]
            is None
        )
    assert report["test_judgments_accessed"] is False
    wrong = development.with_columns(pl.lit("test").alias("partition"))
    dense = load_bundle(bundle, encoder.recipe)[0]
    with pytest.raises(ValueError, match="development"):
        compare_development(
            wrong,
            load_lexical(lexical),
            dense,
            encoder,
            tmp_path / "bad",
            {"recipe_id": digest(encoder.recipe)},
        )
    assert not (tmp_path / "bad").exists()
    with pytest.raises(InvalidDense):
        evaluate_development(
            prepared,
            lexical,
            bundle,
            encoder,
            tmp_path / "bad-pin",
            {**pins, "development_sha256": "wrong"},
            job,
        )
