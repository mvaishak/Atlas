"""Independent cosine/RRF fixtures; no actual model, Hub access, or acquired data."""

from pathlib import Path

import faiss
import numpy as np
import pytest

from atlasrank.evaluation.vector_checks import compare_reference_ranking
from atlasrank.retrieval.dense import DenseIndex
from atlasrank.retrieval.encoder import configuration, validate_vectors, verify_snapshot
from atlasrank.retrieval.fusion import reciprocal_rank_fusion


def test_exact_cosine_negative_scores_native_save_load(tmp_path: Path) -> None:
    vectors = np.array([[1, 0], [0.6, 0.8], [0, 1], [-1, 0]], dtype=np.float32)
    keys = [("us", str(i)) for i in range(4)]
    query = np.array([[1, 0]], dtype=np.float32)
    index = DenseIndex(keys, vectors)
    result = index.search(query, 100)
    assert [key for key, _ in result] == keys
    np.testing.assert_allclose(
        [s for _, s in result], (vectors @ query.T).ravel(), rtol=1e-5, atol=1e-6
    )
    # Negative similarity still occupies its catalog position, with no BM25 positive-only filter.
    assert result[-1][1] == -1
    path = tmp_path / "fixture.faiss"
    faiss.write_index(index.index, str(path))
    restored = faiss.read_index(str(path))
    assert isinstance(restored, faiss.IndexFlatIP)
    np.testing.assert_array_equal(restored.reconstruct_n(0, 4), vectors)
    index.index = restored
    assert index.search(query, 100) == result


def test_cutoff_ties_include_all_canonical_ids() -> None:
    keys = [("us", f"{i:03}") for i in range(105)]
    vectors = np.tile(np.array([[1, 0]], dtype=np.float32), (105, 1))
    index = DenseIndex(keys, vectors)
    assert [key for key, _ in index.search(vectors[:1], 100)] == keys[:100]
    assert index.search(vectors[:1], 5) == [(key, 1) for key in keys[:5]]
    with pytest.raises(ValueError):
        DenseIndex(keys[::-1], vectors)
    with pytest.raises(ValueError):
        DenseIndex(keys[:-1] + [keys[0]], vectors)


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0, 0]], dtype=np.float32),
        np.array([[float("nan"), 1]], dtype=np.float32),
        np.array([[float("inf"), 1]], dtype=np.float32),
        np.array([[1, 0]], dtype=np.float64),
        np.array([1, 0], dtype=np.float32),
    ],
)
def test_vector_validation_rejects_invalid_norm_dtype_shape(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_vectors(values, 2)


def test_query_bounds_and_alignment_failures() -> None:
    index = DenseIndex([("us", "a")], np.array([[1, 0]], dtype=np.float32))
    for k in [0, 101, True]:
        with pytest.raises(ValueError):
            index.search(np.array([[1, 0]], dtype=np.float32), k)
    for query in [
        np.array([[1, 0, 0]], dtype=np.float32),
        np.array([[1, 0], [1, 0]], dtype=np.float32),
    ]:
        with pytest.raises(ValueError):
            index.search(query)


def test_rrf_hand_arithmetic_absent_lists_and_ties() -> None:
    a, b, c = [("us", x) for x in "abc"]
    result = reciprocal_rank_fusion([[a, b], [c, b]])
    assert result == [(b, 2 / 62), (a, 1 / 61), (c, 1 / 61)]
    assert reciprocal_rank_fusion([[a, b], []], 1) == [(a, 1 / 61)]
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([[c, b], [a, b]]) == result
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[a, a]])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[a]], True)


def test_frozen_config_and_missing_snapshot_fail_closed(tmp_path: Path) -> None:
    import json

    config = Path(__file__).parent.parent / "configs/dense.json"
    spec = configuration(config)
    spec["model_path"] = str(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(spec)
    spec["device"] = "mps"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        configuration(path)


def test_calibration_requires_approval_limits_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from types import SimpleNamespace

    import psutil

    import atlasrank.retrieval.calibration as job

    spec = json.loads((Path(__file__).parent.parent / "configs/dense.json").read_text())
    spec["model_path"] = str(tmp_path / "model")
    spec["prepared_path"] = str(tmp_path / "prepared")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=3 * 1024**3))
    with pytest.raises(RuntimeError, match="4 GiB"):
        job.calibrate(config, tmp_path, tmp_path / "pilot")
    assert not (tmp_path / "pilot").exists()


@pytest.mark.parametrize("damage", ["metadata", "weights", "extra"])
def test_snapshot_integrity_rejects_changes(tmp_path: Path, damage: str) -> None:
    import hashlib

    from atlasrank.retrieval.encoder import MODEL_FILES

    entries = []
    for name in sorted(MODEL_FILES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b"synthetic fixture content"
        path.write_bytes(data)
        entry = {"name": name, "bytes": len(data)}
        if name == "model.safetensors":
            entry["sha256"] = hashlib.sha256(data).hexdigest()
        else:
            entry["git_blob"] = hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data
            ).hexdigest()
        entries.append(entry)
    spec = {"model_path": str(tmp_path), "files": entries}
    assert len(verify_snapshot(spec)) == 11
    if damage == "extra":
        (tmp_path / "unapproved.bin").write_bytes(b"unapproved")
    else:
        name = "README.md" if damage == "metadata" else "model.safetensors"
        (tmp_path / name).write_bytes(b"Xynthetic fixture content")
    with pytest.raises(ValueError):
        verify_snapshot(spec)


def test_label_free_sample_is_hash_selected_and_order_independent(tmp_path: Path) -> None:
    import hashlib

    import polars as pl

    from atlasrank.retrieval.calibration import sample_catalog

    frame = pl.DataFrame(
        {
            "product_locale": ["us"] * 8,
            "product_id": list("abcdefgh"),
            "product_title": ["synthetic title"] * 8,
        }
    )
    frame.write_parquet(tmp_path / "a.parquet")
    frame.reverse().write_parquet(tmp_path / "b.parquet")
    a = sample_catalog(tmp_path / "a.parquet", 4)
    b = sample_catalog(tmp_path / "b.parquet", 4)
    assert a.equals(b)
    expected = sorted(
        sorted("abcdefgh", key=lambda p: hashlib.sha256(f"42|us|{p}".encode()).hexdigest())[:4]
    )
    assert a["product_id"].to_list() == expected


def test_cross_kernel_near_ties_keep_existing_tolerance_and_actual_tie_rule() -> None:
    reference = np.array([0.5, np.nextafter(np.float32(0.5), np.float32(1)), 0], dtype=np.float32)
    result = compare_reference_ranking(reference, [0, 1], [0.5, 0.5], 2)
    assert result["membership_exact"] and not result["order_exact"]
    assert result["rtol"] == 1e-5 and result["atol"] == 1e-6
    with pytest.raises(AssertionError, match="ties"):
        compare_reference_ranking(reference, [1, 0], [0.5, 0.5], 2)
    with pytest.raises(AssertionError):
        compare_reference_ranking(reference, [0, 1], [0.6, 0.5], 2)
    with pytest.raises(AssertionError, match="better reference candidate"):
        compare_reference_ranking(
            np.array([0.9, 0.2, 0.1], dtype=np.float32), [1, 2], [0.2, 0.1], 2
        )
