"""Learned-model loading must reject corrupt bytes before native execution."""

import sys
from dataclasses import asdict
from pathlib import Path

import pytest


def test_corrupt_learned_artifact_is_rejected_before_native_import(tmp_path: Path) -> None:
    from test_ranking_parity import configured

    from atlasrank.artifacts.dense import InvalidDense, seal
    from atlasrank.ranking.learned import XGBoostRanker

    context, _, _, _ = configured(tmp_path)
    artifact = tmp_path / "learned"
    artifact.mkdir()
    (artifact / "model.ubj").write_bytes(b"not-native-executable")
    seal(
        artifact,
        {
            "schema_version": 1,
            "kind": "atlasrank.xgboost_ranker",
            "provenance": asdict(context.provenance),
        },
        {"model.ubj"},
    )
    (artifact / "model.ubj").chmod(0o644)
    (artifact / "model.ubj").write_bytes(b"changed")
    before = "xgboost" in sys.modules
    with pytest.raises(InvalidDense):
        XGBoostRanker.load(artifact, context.provenance)
    assert ("xgboost" in sys.modules) == before
