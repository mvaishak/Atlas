"""Small historical M5 contract fixtures; no model, GPU or private data."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from atlasrank.artifacts.dense import file_hash
from atlasrank.evaluation.m5 import compare_pairs, evaluate_rankings, validate_development
from atlasrank.retrieval.m5_encoder import verify_file, verify_model


def development() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": [1, 1, 2],
            "query": ["Ａ  B", "Ａ  B", "other"],
            "product_locale": ["us"] * 3,
            "product_id": ["a", "b", "b"],
            "esci_label": ["E", "I", "I"],
            "partition": ["development"] * 3,
        }
    )


def test_identity_and_corrupt_artifact(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"fixture, not a real model")
    pins = {"model.safetensors": file_hash(weights)}
    verify_model(model, pins)
    weights.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        verify_model(model, pins)
    with pytest.raises(FileNotFoundError):
        verify_file(tmp_path / "absent", "0" * 64)
    with pytest.raises(ValueError):
        verify_model(model, {"../escape": "0" * 64})


def test_development_query_isolation() -> None:
    groups = validate_development(development())
    assert groups[1] == "a b"
    with pytest.raises(ValueError):
        validate_development(development().with_columns(pl.lit("train").alias("partition")))
    with pytest.raises(ValueError):
        validate_development(pl.concat([development(), development().head(1)]))
    bad = development().with_columns(pl.Series("query", ["one", "two", "other"]))
    with pytest.raises(ValueError):
        validate_development(bad)


def test_track_b_unknowns_short_lists_and_rrf() -> None:
    bm25 = {1: [("us", "b")], 2: []}
    dense = {1: [("us", "z"), ("us", "a")], 2: []}
    catalog = {("us", x) for x in ["a", "b", "z"]}
    rows, summary = evaluate_rankings(development(), bm25, dense, catalog)
    assert rows[1]["dense"]["known_exact_recall_at_20"] == 1.0
    assert rows[2]["dense"]["known_exact_recall_at_20"] is None
    assert summary["dense"]["judgment_coverage_at_10"] == 0.25
    assert summary["bm25"]["returned_at_100_total"] == 1
    assert summary["dense"]["recall_excluded_queries"] == 1
    from atlasrank.retrieval.fusion import reciprocal_rank_fusion

    fused = [k for k, _ in reciprocal_rank_fusion([bm25[1], dense[1]])]
    assert fused == [("us", "b"), ("us", "z"), ("us", "a")]
    with pytest.raises(ValueError):
        evaluate_rankings(development(), {1: []}, dense, catalog)
    with pytest.raises(ValueError):
        evaluate_rankings(development(), bm25, {1: [("us", "bad")], 2: []}, catalog)


def test_paired_matching_and_grouped_bootstrap() -> None:
    rows = [{"query_id": i, "group": "same", "reference": 0.0, "variant": 0.25} for i in [1, 2]]
    result = compare_pairs(rows)
    assert result["paired_interval"]["groups"] == 1
    assert result["paired_interval"]["lower"] == 0.25
    assert result["queries_improved"] == 2
    with pytest.raises(ValueError):
        compare_pairs(rows + rows[:1])
    with pytest.raises(ValueError):
        compare_pairs([{**rows[0], "variant": np.nan}])


def test_track_a_and_short_bm25_export(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from atlasrank.evaluation.m5 import export_bm25, read_rankings, track_a

    class Encoder:
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.array(
                [[1, 0] if text in {"Ａ  B", "positive"} else [0, 1] for text in texts],
                dtype=np.float32,
            )

    class Lexical:
        def score_all(self, query: str) -> str:
            return query

        def hits_from_scores(self, scores: str, k: int) -> list[SimpleNamespace]:
            assert k == 100
            return (
                [SimpleNamespace(product_locale="us", product_id="a")] if scores == "Ａ  B" else []
            )

    products = pl.DataFrame(
        {
            "product_locale": ["us", "us"],
            "product_id": ["a", "b"],
            "product_title": ["positive", "negative"],
        }
    )
    rows, summary = track_a(development(), products, Encoder())
    assert summary["ndcg_at_10"] == 0.5  # zero-IDCG second query remains in macro mean
    assert summary["mrr_at_10"] == 0.5
    assert len(rows) == 2 and "known_exact_recall_at_100" not in summary
    path = tmp_path / "bm25.jsonl"
    assert export_bm25(development(), Lexical(), path) == {"queries": 2, "returned_hits": 1}
    assert read_rankings(path) == {1: [("us", "a")], 2: []}
    with pytest.raises(FileExistsError):
        export_bm25(development(), Lexical(), path)


def test_historical_catalog_vs_fixed_query_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace
    from typing import Any

    from atlasrank.retrieval.m5_encoder import M5Encoder

    captured: list[Any] = []

    class FakeModel:
        def eval(self) -> None:
            pass

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            captured.append((texts, kwargs))
            result = np.zeros((len(texts), 384), dtype=np.float32)
            result[:, 0] = 1
            return result

    class FakeShared:
        def __init__(self, path: str, device: str) -> None:
            self.model = FakeModel()

    module = ModuleType("atlasrank.training.model")
    module.SharedMiniLM = FakeShared  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.__version__ = "fixture"  # type: ignore[attr-defined]
    torch.version = SimpleNamespace(cuda=None)  # type: ignore[attr-defined]
    torch.set_num_threads = lambda _: None  # type: ignore[attr-defined]
    torch.get_num_threads = lambda: 1  # type: ignore[attr-defined]
    torch.autocast = lambda **_: nullcontext()  # type: ignore[attr-defined]
    torch.inference_mode = nullcontext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atlasrank.training.model", module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        "atlasrank.retrieval.m5_encoder.importlib.metadata.version", lambda _: "6.1.0"
    )
    monkeypatch.setattr("atlasrank.retrieval.m5_encoder.verify_model", lambda *_: None)
    encoder = M5Encoder(
        tmp_path, {"model_files": {}, "base_revision": "fixture"}, device="cpu", catalog=True
    )
    assert encoder.recipe["batch_size"] == 1024 and encoder.recipe["model_batch_size"] == 32
    encoder.encode([" ＦOO\tBar "])
    assert captured[0][0] == ["foo bar"]
    assert captured[0][1]["normalize_embeddings"] is True
    assert captured[0][1]["batch_size"] == 32
    with pytest.raises(ValueError):
        encoder.encode([])


def test_recipe_substitution_fails() -> None:
    import json

    from atlasrank.retrieval.m5_encoder import verify_recipe

    pins = json.loads(Path("configs/m5-reproduction.json").read_text())
    with pytest.raises(ValueError, match="recipe"):
        verify_recipe({"model_files": pins["model_files"], "max_length": 128}, pins)
