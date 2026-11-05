"""Immutable learned XGBoost adapter using the existing exact native boundary."""

import importlib.metadata
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlasrank.artifacts.dense import digest, validate
from atlasrank.artifacts.query_recipe import validate_query_recipe
from atlasrank.features.ranking import SCHEMA, SCHEMA_ID, Provenance
from atlasrank.ranking.experiment_contract import fixed_configuration
from atlasrank.ranking.prediction import DeterministicRanker, PredictionInput
from atlasrank.ranking.xgboost_boundary import ZeroUpdateModel, capture_native


class FittedPredictor(ZeroUpdateModel):
    """Same DMatrix conversion/capture as the verified native fixture adapter."""

    trained = True

    def __init__(self, model_file: Path) -> None:
        super().__init__()
        self.booster = self.xgb.Booster(model_file=model_file)
        self.booster.set_param({"nthread": 1, "device": "cpu"})
        if self.booster.num_boosted_rounds() != 100 or self.booster.num_features() != 5:
            raise ValueError("Expected the single fixed 100-round five-feature model")
        self.record = False

    def predict(self, inputs: PredictionInput) -> NDArray[np.float32]:
        if self.booster is None:
            raise ValueError("Fitted model unavailable")
        native = self.convert(inputs)
        captured = capture_native(native, inputs)
        if self.record:
            self.observations.append(captured)
        prediction = self.booster.predict(native.matrix, validate_features=True, strict_shape=True)
        return prediction.reshape(-1)


class XGBoostRanker(DeterministicRanker):
    """Reuse final-input checks and offline/HTTP interfaces; never a default pipeline."""

    def __init__(self, provenance: Provenance, artifact_id: str, model_file: Path) -> None:
        # The fixture constructor's prohibition on trained models remains untouched.
        self.provenance, self.artifact_id, self.observe = provenance, artifact_id, None
        self.model = FittedPredictor(model_file)
        self.validate(provenance)

    def validate(self, provenance: Provenance) -> None:
        if (
            self.provenance != provenance
            or provenance.schema_id != SCHEMA_ID
            or not self.model.trained
        ):
            raise ValueError("Incompatible learned model or feature provenance")

    @classmethod
    def load(
        cls,
        directory: Path,
        expected: Provenance,
        observe: Any = None,
        *,
        reference_query_recipe: dict[str, Any] | None = None,
        runtime_query_recipe: dict[str, Any] | None = None,
    ) -> "XGBoostRanker":
        metadata = validate(directory, "atlasrank.xgboost_ranker", {"model.ubj"})
        provenance = dict(metadata["provenance"])
        if reference_query_recipe is not None:
            if (
                runtime_query_recipe is None
                or digest(runtime_query_recipe) != expected.query_recipe_id
            ):
                raise ValueError("Actual query recipe provenance differs")
            validate_query_recipe(
                runtime_query_recipe, provenance["query_recipe_id"], reference_query_recipe
            )
            # Comparison only. The sealed metadata and its original digest remain intact.
            provenance["query_recipe_id"] = expected.query_recipe_id
        elif runtime_query_recipe is not None:
            raise ValueError("A reference query recipe is required for portability")
        if (
            provenance != asdict(expected)
            or metadata["feature_schema"] != SCHEMA
            or metadata["feature_schema_id"] != SCHEMA_ID
            or metadata["xgboost_version"] != importlib.metadata.version("xgboost-cpu")
        ):
            raise ValueError("Incompatible learned model schema/source/runtime")
        fixed_configuration(metadata["configuration"])
        result = cls(expected, metadata["artifact_id"], directory / "model.ubj")
        result.observe = observe
        return result
