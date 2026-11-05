"""Golden lexical math, stable identity selection, and bundle failure boundaries."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from atlasrank.artifacts.lexical import InvalidIndex, seal
from atlasrank.data.common import write_json
from atlasrank.retrieval.analyzer import analyze
from atlasrank.retrieval.bm25 import build, load, top_rows


def golden() -> tuple[pl.DataFrame, list[dict[str, object]]]:
    fixture = json.loads((Path(__file__).parent / "fixtures/bm25.json").read_text())
    return pl.DataFrame(fixture["documents"]), fixture["queries"]


def test_golden_independent_lucene_math_and_save_load(tmp_path: Path) -> None:
    products, queries = golden()
    products = products.with_columns(
        [
            pl.lit("brandonlyneedle").alias(name)
            for name in (
                "product_description",
                "product_bullet_point",
                "product_brand",
                "product_color",
            )
        ]
    )
    report = build(products, tmp_path / "index", {"kind": "synthetic"})
    index = load(tmp_path / "index")
    assert len(index.keys) == 4  # The empty-title product remains in N and mapping.
    assert report["probe_hashes_before_load"] == index.probe_hashes()
    for case in queries:
        hits = index.search(str(case["query"]))
        assert [hit.product_id for hit in hits] == case["expected_ids"]
    # Hand reference: N=4, df(apple)=2, avgdl=7/4, no (k1+1) factor.
    scores = index.score_all("apple")
    idf = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))
    a = idf * 2 / (2 + 1.5 * (0.25 + 0.75 * 3 / 1.75))
    b = idf / (1 + 1.5 * (0.25 + 0.75 * 2 / 1.75))
    np.testing.assert_allclose(scores, [a, b, 0, 0], rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(index.score_all("apple apple"), 2 * scores)
    assert analyze("Straße_ＦＯＯ / café 12.5") == ["strasse", "foo", "café", "12", "5"]
    assert index.search("brandonlyneedle") == []
    with pytest.raises(FileExistsError):
        build(products, tmp_path / "index", {})


def test_reordered_build_identical_and_ties_cross_top_k_boundary(tmp_path: Path) -> None:
    products = pl.DataFrame(
        {
            "product_locale": ["us"] * 105,
            "product_id": [f"{i:03d}" for i in reversed(range(105))],
            "product_title": ["same token"] * 105,
        }
    )
    a = build(products, tmp_path / "a", {"kind": "synthetic"})
    b = build(products.reverse(), tmp_path / "b", {"kind": "synthetic"})
    assert a["bundle_id"] == b["bundle_id"]
    index = load(tmp_path / "a")
    assert [hit.product_id for hit in index.search("same", 100)] == [f"{i:03d}" for i in range(100)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda _: index.search("same", 10), range(12)))
    assert all(response == index.search("same", 10) for response in responses)
    np.testing.assert_array_equal(
        top_rows(np.array([0, 2, 2, 3, 2], dtype=np.float32), 3), [3, 1, 2]
    )
    for k in (0, 101, True):
        with pytest.raises(ValueError):
            index.search("same", k)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "corrupt",
        "schema",
        "analyzer",
        "version",
        "params",
        "mapping",
        "vocab",
        "shape",
        "parquet",
    ],
)
def test_corrupt_or_incompatible_bundle_fails_closed(tmp_path: Path, damage: str) -> None:
    products, _ = golden()
    bundle = tmp_path / "index"
    build(products, bundle, {"kind": "synthetic"})
    manifest = json.loads((bundle / "manifest.json").read_text())
    metadata = {k: v for k, v in manifest.items() if k not in {"files", "bundle_id"}}
    if damage == "missing":
        (bundle / "backend/data.csc.index.npy").rename(bundle / "removed.npy")
    elif damage == "corrupt":
        with (bundle / "backend/data.csc.index.npy").open("ab") as out:
            out.write(b"corrupt")
    elif damage in {"schema", "analyzer", "version"}:
        key = {"schema": "schema_version", "analyzer": "analyzer", "version": "bm25s_version"}[
            damage
        ]
        metadata[key] = 999 if damage == "schema" else "wrong"
        seal(bundle, metadata)
    elif damage == "params":
        p = bundle / "backend/params.index.json"
        params = json.loads(p.read_text())
        params["b"] = 0.1
        write_json(p, params)
        seal(bundle, metadata)
    elif damage == "mapping":
        products.head(1).write_parquet(bundle / "products.parquet")
        seal(bundle, metadata)
    elif damage == "vocab":
        p = bundle / "backend/vocab.index.json"
        vocab = json.loads(p.read_text())
        vocab[next(iter(vocab))] = 999
        write_json(p, vocab)
        seal(bundle, metadata)
    elif damage == "shape":
        p = bundle / "backend/data.csc.index.npy"
        values = np.load(p, allow_pickle=False)
        np.save(p, values.reshape(1, -1), allow_pickle=False)
        seal(bundle, metadata)
    else:
        (bundle / "products.parquet").write_bytes(b"invalid parquet")
        seal(bundle, metadata)
    with pytest.raises(InvalidIndex):
        load(bundle)


def test_invalid_catalog_and_candidate_alignment(tmp_path: Path) -> None:
    products, _ = golden()
    for invalid in (
        products.head(0),
        pl.concat([products, products.head(1)]),
        products.with_columns(pl.lit("jp").alias("product_locale")),
        products.with_columns(pl.lit("").alias("product_title")),
    ):
        with pytest.raises(ValueError):
            build(invalid, tmp_path / "invalid", {})
    build(products, tmp_path / "valid", {})
    index = load(tmp_path / "valid")
    with pytest.raises(ValueError):
        index.candidate_scores(index.score_all("apple"), [("us", "absent")])
    with pytest.raises(ValueError):
        index.hits_from_scores(np.zeros(1, dtype=np.float32), 10)


@pytest.mark.parametrize("all_irrelevant", [False, True])
def test_end_to_end_development_job_and_no_exact_denominators(
    tmp_path: Path, all_irrelevant: bool
) -> None:
    from atlasrank.data.common import sha256
    from atlasrank.retrieval.jobs import build_job, evaluate_job

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (tmp_path / "execution").mkdir()
    products, _ = golden()
    products.write_parquet(prepared / "catalog.parquet")
    pl.DataFrame(
        {
            "query_id": [1, 2],
            "query": ["apple", "berry"],
            "product_locale": ["us", "us"],
            "product_id": ["a", "c"],
            "esci_label": ["I" if all_irrelevant else "E", "I"],
            "group_sha256": ["g1", "g2"],
        }
    ).write_parquet(prepared / "development.parquet")
    write_json(
        prepared / "audit.json",
        {
            "schema_version": 1,
            "evaluation_contract": "v0.3",
            "output_sha256": {"development.parquet": sha256(prepared / "development.parquet")},
        },
    )
    write_json(
        prepared / "COMPLETE.json",
        {"schema_version": 1, "audit_sha256": sha256(prepared / "audit.json")},
    )
    config = tmp_path / "config.json"
    spec = json.loads((Path(__file__).parent.parent / "configs/bm25.json").read_text())
    spec.update(
        prepared_path=str(prepared),
        index_path=str(tmp_path / "index"),
        prepared_audit_sha256=sha256(prepared / "audit.json"),
        catalog_sha256=sha256(prepared / "catalog.parquet"),
        development_sha256=sha256(prepared / "development.parquet"),
    )
    write_json(config, spec)
    build_job(config, tmp_path, tmp_path / "index")
    output = evaluate_job(config, tmp_path, tmp_path / "report")
    report = json.loads((output / "report.json").read_text())
    assert report["A"]["queries"] == 2
    assert report["B"]["recall_queries"] == (0 if all_irrelevant else 1)
    assert report["B"]["recall_excluded_queries"] == (2 if all_irrelevant else 1)
    assert report["test_labels_accessed"] is False
    if all_irrelevant:
        assert report["query_sampling_intervals"]["B.known_exact_recall_at_100"]["mean"] is None
