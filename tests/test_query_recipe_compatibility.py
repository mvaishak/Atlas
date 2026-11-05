"""Platform portability preserves every scoring and dependency pin."""

import importlib.metadata
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from atlasrank.artifacts.dense import digest


def recipes() -> tuple[dict[str, Any], dict[str, Any]]:
    reference: dict[str, Any] = {
        "runtime_platform": {"platform": "Linux-original-glibc2.39", "machine": "x86_64"},
        "runtime_versions": {"torch": "2.14.1+cpu", "polars": "2.1.0"},
        "encoder_code_sha256": "a" * 64,
        "model_files_sha256": {"weights": "b" * 64},
        "pooling": "attention_mask_mean_then_l2",
    }
    actual = deepcopy(reference)
    actual["runtime_platform"]["platform"] = "Linux-dsmlp-glibc2.35"
    return reference, actual


def test_platform_difference_requires_authenticated_reference() -> None:
    from atlasrank.artifacts.query_recipe import validate_query_recipe

    reference, actual = recipes()
    validate_query_recipe(reference, digest(reference))
    with pytest.raises(ValueError):
        validate_query_recipe(actual, digest(reference))
    validate_query_recipe(actual, digest(reference), reference)
    with pytest.raises(ValueError):
        validate_query_recipe(actual, "0" * 64, reference)
    assert digest(actual) != digest(reference)


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_versions", {"torch": "other", "polars": "2.1.0"}),
        ("encoder_code_sha256", "c" * 64),
        ("model_files_sha256", {"weights": "d" * 64}),
        ("pooling", "other"),
        ("runtime_platform", {"platform": "Linux-dsmlp", "machine": "aarch64"}),
        ("runtime_platform", {"platform": "Darwin-host", "machine": "x86_64"}),
    ],
)
def test_portability_rejects_non_platform_drift(field: str, value: Any) -> None:
    from atlasrank.artifacts.query_recipe import validate_query_recipe

    reference, actual = recipes()
    actual[field] = value
    with pytest.raises(ValueError):
        validate_query_recipe(actual, digest(reference), reference)


def test_learned_loader_binds_actual_recipe_without_changing_sealed_metadata(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from test_ranking_parity import configured

    from atlasrank.features.ranking import SCHEMA, SCHEMA_ID
    from atlasrank.ranking import learned
    from atlasrank.ranking.experiment_contract import fixed_configuration

    context, _, _, _ = configured(tmp_path)
    reference, actual = recipes()
    historical = replace(context.provenance, query_recipe_id=digest(reference))
    expected = replace(context.provenance, query_recipe_id=digest(actual))
    metadata = {
        "provenance": asdict(historical),
        "feature_schema": SCHEMA,
        "feature_schema_id": SCHEMA_ID,
        "xgboost_version": "3.1.1",
        "configuration": fixed_configuration(),
        "artifact_id": "sealed-model",
    }
    original = deepcopy(metadata)
    monkeypatch.setattr(learned, "validate", lambda *args: metadata)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.1.1")

    def constructor(self: Any, provenance: Any, identity: str, model_file: Path) -> None:
        self.provenance = provenance
        self.artifact_id = identity

    monkeypatch.setattr(learned.XGBoostRanker, "__init__", constructor)
    with pytest.raises(ValueError, match="schema/source/runtime"):
        learned.XGBoostRanker.load(tmp_path, expected)
    loaded = learned.XGBoostRanker.load(
        tmp_path, expected, reference_query_recipe=reference, runtime_query_recipe=actual
    )
    assert loaded.provenance == expected
    assert metadata == original
    with pytest.raises(ValueError, match="schema/source/runtime"):
        learned.XGBoostRanker.load(
            tmp_path,
            replace(expected, catalog_sha256="changed"),
            reference_query_recipe=reference,
            runtime_query_recipe=actual,
        )
