"""Synthetic contracts only. No routine test reads the acquired ESCI files."""

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from atlasrank.data.acquire import acquire
from atlasrank.data.common import check_disk, sha256
from atlasrank.data.prepare import TEXT_FIELDS, load_judgments, prepare
from atlasrank.data.splits import Query, assign_queries, normalize_query
from atlasrank.data.supervision import admissible, validate_pairs


def synthetic(root: Path, fault: str = "") -> Path:
    revision = "a" * 40
    raw = root / "raw" / revision
    raw.mkdir(parents=True)
    examples = []
    for i in range(1, 31):
        examples.append(
            {
                "example_id": i,
                "query_id": i,
                "query": f"query {i}",
                "product_id": "a",
                "product_locale": "us",
                "small_version": 1,
                "split": "train",
                "esci_label": "E" if i % 2 else "I",
            }
        )
    # Deliberately invalid held-out label: preparation must never inspect it.
    examples.append(
        {
            **examples[0],
            "example_id": 100,
            "query_id": 100,
            "query": "held out",
            "split": "test",
            "esci_label": "sealed_not_a_valid_label",
        }
    )
    examples.append({**examples[0], "example_id": 101})
    if fault == "conflict":
        examples[-1]["esci_label"] = "S"
    elif fault == "label":
        examples[0]["esci_label"] = "bad"
    elif fault == "join":
        examples[0]["product_id"] = "missing"
    elif fault == "query":
        examples[0]["query"] = " "
    elif fault == "null":
        examples[0]["query"] = None
    elif fault == "overlap":
        examples[0]["query"] = "held out"
    pl.DataFrame(examples).write_parquet(raw / "examples.parquet", row_group_size=10)
    products: dict[str, list[str | None]] = {
        "product_id": ["a", "unjudged", "foreign"],
        "product_locale": ["us", "us", "es"],
        **{field: [None, "available", "other"] for field in TEXT_FIELDS},
    }
    frame = pl.DataFrame(products)
    if fault == "product_duplicate":
        frame = pl.concat([frame, frame.head(1)])
    frame.write_parquet(raw / "products.parquet", row_group_size=1)
    spec = {
        "schema_version": 1,
        "revision": revision,
        "files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(raw.glob("*.parquet"))
        ],
    }
    config = root / "config.json"
    config.write_text(json.dumps(spec))
    return config


def test_connected_groups_transitive_quarantine_and_order_invariance() -> None:
    train = [
        Query("us", 1, " Alpha "),
        Query("us", 1, "Bridge"),
        Query("us", 2, "ＢＲＩＤＧＥ"),
        Query("us", 3, "other"),
        Query("es", 1, "Alpha"),
        Query("us", 4, "QUERY"),
        Query("us", 4, "query"),
    ]
    test = [Query("us", 99, "alpha")]
    first = assign_queries(train, test)
    assert first == assign_queries(reversed(train), reversed(test))
    assert all(
        a.partition == "quarantine" for a in first if a.query_id in {1, 2} and a.locale == "us"
    )
    assert all(a.group_key == "alpha" for a in first if a.partition == "quarantine")
    assert all(a.partition != "quarantine" for a in first if a.locale == "es")
    assert normalize_query("  ＦＯＯ\tStraße\n") == "foo strasse"
    for a in first:
        if a.partition != "quarantine":
            digest = int(hashlib.sha256(f"42|{a.locale}|{a.group_key}".encode()).hexdigest(), 16)
            assert (a.partition == "development") == (digest < (2**256) / 5)
    with pytest.raises(ValueError):
        assign_queries([Query("us", 1, " ")], [])


def test_prepare_is_reproducible_complete_and_keeps_test_labels_sealed(tmp_path: Path) -> None:
    config = synthetic(tmp_path)
    a, b = tmp_path / "prepared-a", tmp_path / "prepared-b"
    prepare(config, tmp_path, a)
    prepare(config, tmp_path, b)
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {
        p.name: p.read_bytes() for p in b.iterdir()
    }
    audit = json.loads((a / "audit.json").read_text())
    assert audit["examples"]["identical_duplicate_train_pairs_collapsed"] == 1
    assert audit["examples"]["test_labels_accessed_by_application"] is False
    assert audit["examples"]["official_test_rows_preserved"] == 1
    assert audit["catalog"]["us_unique_products"] == 2
    catalog = pl.read_parquet(a / "catalog.parquet")
    assert catalog.filter(pl.col("product_id") == "a")["product_title"][0] == ""
    assert "esci_label" not in pl.read_parquet(a / "test.references.parquet").columns
    assert "esci_label" not in pl.read_parquet(a / "quarantine.parquet").columns
    train = load_judgments(a, "train")
    dev = load_judgments(a, "development")
    assert not set(train["group_key"]).intersection(dev["group_key"])
    assert train.height + dev.height == 30
    for forbidden in ("test", "test.references", "quarantine", "../raw"):
        with pytest.raises(ValueError):
            load_judgments(a, forbidden)
    with pytest.raises(FileExistsError):
        prepare(config, tmp_path, a)
    with (a / "train.parquet").open("ab") as out:
        out.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_judgments(a, "train")


@pytest.mark.parametrize(
    "fault", ["conflict", "label", "join", "query", "null", "product_duplicate"]
)
def test_preparation_fails_closed_and_preserves_raw(tmp_path: Path, fault: str) -> None:
    config = synthetic(tmp_path, fault)
    paths = list((tmp_path / "raw").rglob("*.parquet"))
    before = {p: sha256(p) for p in paths}
    with pytest.raises(ValueError):
        prepare(config, tmp_path, tmp_path / "invalid")
    assert before == {p: sha256(p) for p in paths}
    assert not (tmp_path / "invalid/COMPLETE.json").exists()


def test_source_checksum_mismatch_prevents_preparation(tmp_path: Path) -> None:
    config = synthetic(tmp_path)
    spec = json.loads(config.read_text())
    spec["files"][0]["sha256"] = "0" * 64
    config.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="mismatch"):
        prepare(config, tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_overlap_is_quarantined_in_actual_manifests(tmp_path: Path) -> None:
    config = synthetic(tmp_path, "overlap")
    out = tmp_path / "prepared"
    prepare(config, tmp_path, out)
    quarantined = pl.read_parquet(out / "quarantine.parquet")
    assert quarantined["query_id"].to_list() == [1]
    assert "esci_label" not in quarantined.columns
    for split in ("train", "development"):
        assert 1 not in load_judgments(out, split)["query_id"].to_list()


def test_supervision_truth_table_excludes_unknown_cross_query_and_held_out() -> None:
    judgments = validate_pairs(
        [("e1", "E"), ("e2", "E"), ("i", "I"), ("s", "S"), ("c", "C"), ("i", "I")]
    )
    result = admissible("train", judgments)
    assert asdict(result) == {
        "positives": frozenset({"e1", "e2"}),
        "negatives": frozenset({"i"}),
        "usable": True,
    }
    other_query = admissible("train", {"e1": "I", "other": "E"})
    assert "e1" in other_query.negatives and "e1" not in result.negatives
    assert not result.negatives.intersection({"unknown", "e1", "e2", "s", "c", "other"})
    assert not admissible("train", {"a": "E"}).usable
    assert not admissible("train", {"a": "I"}).usable
    assert not admissible("train", {}).usable
    for split in ("development", "test", "quarantine"):
        with pytest.raises(ValueError):
            admissible(split, judgments)
    with pytest.raises(ValueError):
        validate_pairs([("a", "E"), ("a", "I")])


def test_acquisition_pins_reuses_and_preserves_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic source"

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

    calls = []

    def fetch(*args: Any, **kwargs: Any) -> Response:
        calls.append(args)
        return Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    config = tmp_path / "config.json"
    spec = {
        "schema_version": 1,
        "repository": "https://github.com/amazon-science/esci-data",
        "revision": "a" * 40,
        "files": [
            {
                "name": "fixture.txt",
                "path": "fixture.txt",
                "transport": "raw",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    config.write_text(json.dumps(spec))
    raw = acquire(config, tmp_path / "data")
    acquire(config, tmp_path / "data")
    assert len(calls) == 1
    source = raw / "fixture.txt"
    source.chmod(0o644)
    source.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="preserved"):
        acquire(config, tmp_path / "data")
    assert source.read_bytes() == b"corruption"


def test_disk_limit_failure_before_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("atlasrank.data.common.DISK_LIMIT", 10)
    with pytest.raises(RuntimeError, match="30 GiB"):
        check_disk(tmp_path, additional=11)


def test_bad_download_is_preserved_without_promoting_raw_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        headers: dict[str, str] = {}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"wrong"))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "https://github.com/amazon-science/esci-data",
                "revision": "a" * 40,
                "files": [
                    {
                        "name": "fixture",
                        "path": "fixture",
                        "transport": "raw",
                        "bytes": 5,
                        "sha256": hashlib.sha256(b"right").hexdigest(),
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="partial preserved"):
        acquire(config, tmp_path / "data")
    raw = tmp_path / "data/raw" / ("a" * 40)
    assert not (raw / "fixture").exists()
    assert [p.read_bytes() for p in raw.glob("*.partial")] == [b"wrong"]


def test_monitored_worker_failure_records_evidence(tmp_path: Path) -> None:
    from atlasrank.data.runner import run_bounded

    config = synthetic(tmp_path)
    # Existing destination fails before writes, through the actual worker boundary.
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RuntimeError, match="subprocess failed"):
        run_bounded("prepare", config, tmp_path, output)
    records = list((tmp_path / "execution").glob("prepare-*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["status"] == "failed"
    assert list(output.iterdir()) == []
