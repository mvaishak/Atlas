"""Hand-derived five-feature fixtures, never a learned-model experiment."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from atlasrank.artifacts.dense import digest
from atlasrank.retrieval.bm25 import build, load
from atlasrank.retrieval.dense import DenseIndex


def runtime(root: Path) -> Any:
    from atlasrank.features.ranking import DenseBinding, FeatureContext, model_identity

    class Encoder:
        recipe = {
            "dimensions": 3,
            "batch_size": 32,
            "revision": "synthetic-v1",
            "model_files_sha256": {"model.safetensors": digest("synthetic-model")},
        }

        def encode(self, texts: list[str]) -> Any:
            return np.array([[1, 0, 0]] * len(texts), dtype=np.float32)

    products = pl.DataFrame(
        {
            "product_locale": ["us"] * 4,
            "product_id": ["a", "b", "c", "d"],
            "product_title": ["Red apple apple", "", "blue berry", "!!!"],
        }
    )
    path = root / "lexical"
    build(products, path, {"catalog_sha256": digest("synthetic-catalog")})
    lexical = load(path)
    dense = DenseIndex(
        lexical.keys, np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype=np.float32)
    )
    binding = DenseBinding(
        artifact_id=digest("synthetic-dense"),
        catalog_sha256=digest("synthetic-catalog"),
        catalog_content_id=digest(list(zip(lexical.keys, lexical.titles, strict=True))),
        product_recipe_id=digest(Encoder.recipe),
        model_id=model_identity(Encoder.recipe),
    )
    return FeatureContext(lexical, dense, Encoder(), binding)


def test_hand_calculated_features_and_empty_title_semantics(tmp_path: Path) -> None:
    from atlasrank.features.ranking import FEATURE_NAMES, build_features

    context = runtime(tmp_path)
    batch = build_features(
        context, " ＲＥＤ\t apple apple ", [("us", "c"), ("us", "a"), ("us", "b")]
    )
    assert batch.names == FEATURE_NAMES
    assert batch.keys == (("us", "a"), ("us", "b"), ("us", "c"))
    assert batch.input_rows == (2, 0, 1)
    assert batch.values.dtype == np.float32 and batch.values.flags.c_contiguous
    # Independently hand-checked cosine, unique-token Jaccard and repeated-token lengths.
    assert np.array_equal(
        batch.values[:, 1:],
        np.array([[1, 1, 3, 3], [0, 0, 3, 0], [0.6, 0, 3, 2]], dtype=np.float32),
    )
    assert batch.values[0, 0] > 0 and batch.values[1, 0] == batch.values[2, 0] == 0
    assert batch.missing.dtype == np.bool_ and not batch.missing.any()
    assert np.array_equal(batch.text_empty, [[False, False], [False, True], [False, False]])
    assert not batch.values.flags.writeable
    empty = build_features(context, "", [("us", "b"), ("us", "d")])
    assert np.array_equal(empty.values[:, 2:], np.zeros((2, 3), dtype=np.float32))
    assert np.array_equal(empty.text_empty, [[True, True], [True, False]])


def test_missing_scores_and_identity_mismatch_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atlasrank.features.ranking import build_features

    context = runtime(tmp_path)
    monkeypatch.setattr(context.dense, "candidate_scores", lambda *_: [])
    with pytest.raises(ValueError, match="score"):
        build_features(context, "apple", [("us", "a")])
    with pytest.raises(ValueError):
        build_features(context, "apple", [("us", "absent")])


@pytest.mark.parametrize("fault", ["catalog", "model", "identities"])
def test_incompatible_dense_binding_fails(tmp_path: Path, fault: str) -> None:
    from dataclasses import replace

    from atlasrank.features.ranking import FeatureContext

    context = runtime(tmp_path)
    binding = context.binding
    if fault == "catalog":
        binding = replace(binding, catalog_sha256=digest("wrong"))
    if fault == "model":
        binding = replace(binding, model_id=digest("wrong"))
    if fault == "identities":
        context.dense.keys = tuple(reversed(context.dense.keys))
    with pytest.raises(ValueError):
        FeatureContext(context.lexical, context.dense, context.encoder, binding)


def test_changed_tokenizer_cannot_bind_to_old_product_embeddings(tmp_path: Path) -> None:
    from atlasrank.features.ranking import FeatureContext

    context = runtime(tmp_path)
    context.encoder.recipe["model_files_sha256"]["tokenizer.json"] = digest("different-tokenizer")
    with pytest.raises(ValueError, match="binding"):
        FeatureContext(context.lexical, context.dense, context.encoder, context.binding)


def test_frozen_encoder_owns_its_text_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atlasrank.features.ranking import build_features

    context = runtime(tmp_path)
    original = context.encoder.encode
    seen: list[list[str]] = []

    def encode(texts: list[str]) -> Any:
        seen.append(texts)
        return original(texts)

    monkeypatch.setattr(context.encoder, "encode", encode)
    query = " ＲＥＤ\tapple apple "
    build_features(context, query, [("us", "a")])
    # The frozen encoder must receive its original input, not a new preprocessing path.
    assert seen == [[query]]
