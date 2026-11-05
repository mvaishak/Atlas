"""Exact final-input adapter for an explicit deterministic, non-learned fixture.

No XGBoost/Torch import. Native XGBoost prediction remains a separate Linux gate
following the preserved Mac libomp failure; this adapter does not certify DMatrix.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from atlasrank.artifacts.dense import seal, validate, write_once
from atlasrank.features.ranking import FEATURE_NAMES, SCHEMA, SCHEMA_ID, FeatureBatch, Provenance


@dataclass(frozen=True)
class PredictionInput:
    names: tuple[str, ...]
    feature_types: tuple[str, ...]
    keys: tuple[tuple[str, str], ...]
    values: NDArray[np.float32]
    missing: NDArray[np.bool_]
    text_empty: NDArray[np.bool_]
    query_fingerprint: str
    provenance: Provenance
    schema_id: str

    def __post_init__(self) -> None:
        if self.feature_types != ("float32",) * 5:
            raise ValueError("Final model column dtypes differ")
        checked = FeatureBatch(
            self.names,
            self.keys,
            self.keys,
            self.values,
            self.missing,
            self.text_empty,
            self.query_fingerprint,
            self.provenance,
            self.schema_id,
        )
        for name in ("values", "missing", "text_empty"):
            object.__setattr__(self, name, getattr(checked, name))


def final_input(batch: FeatureBatch) -> PredictionInput:
    # This is the final conversion. This same object is passed directly to model.predict.
    values = batch.values.copy(order="C")
    missing = np.isnan(values)
    empty = batch.text_empty.copy(order="C")
    if not np.array_equal(missing, batch.missing):
        raise ValueError("Final model input missing semantics differ")
    for array in (values, missing, empty):
        array.flags.writeable = False
    return PredictionInput(
        tuple(batch.names),
        (str(values.dtype),) * values.shape[1],
        batch.keys,
        values,
        missing,
        empty,
        batch.query_fingerprint,
        batch.provenance,
        batch.schema_id,
    )


def assert_exact_parity(offline: PredictionInput, serving: PredictionInput) -> None:
    for name in ("names", "feature_types", "keys", "query_fingerprint", "provenance", "schema_id"):
        if getattr(offline, name) != getattr(serving, name):
            raise ValueError(f"Prediction-boundary parity differs: {name}")
    for name in ("values", "missing", "text_empty"):
        a, b = getattr(offline, name), getattr(serving, name)
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"Prediction-boundary parity differs: {name}")


class FixturePredictor(Protocol):
    @property
    def trained(self) -> bool: ...

    def predict(self, inputs: PredictionInput) -> NDArray[np.float32]: ...


class ConstantModel:
    """Explicit zero-coefficient linear fixture, not fitted or a relevance model."""

    trained = False

    def predict(self, inputs: PredictionInput) -> NDArray[np.float32]:
        if (
            inputs.names != FEATURE_NAMES
            or inputs.values.dtype != np.float32
            or inputs.values.shape != (len(inputs.keys), 5)
            or inputs.feature_types != ("float32",) * 5
            or inputs.missing.any()
            or not np.isfinite(inputs.values).all()
        ):
            raise ValueError("Fixture predictor received invalid final inputs")
        return inputs.values @ np.zeros(5, dtype=np.float32) + np.float32(0.5)


def save_deterministic_fixture(output: Path, provenance: Provenance) -> None:
    if output.exists():
        raise FileExistsError("Preserve previous ranker fixtures")
    output.mkdir(parents=True)
    write_once(
        output / "model.json",
        {
            "role": "deterministic_fixture",
            "trained": False,
            "coefficients": [0.0] * 5,
            "bias": 0.5,
            "feature_schema_id": SCHEMA_ID,
        },
    )
    seal(
        output,
        {
            "schema_version": 1,
            "kind": "atlasrank.deterministic_ranker_fixture",
            "feature_schema": SCHEMA,
            "feature_schema_id": SCHEMA_ID,
            "provenance": asdict(provenance),
            "model_role": "deterministic_fixture",
            "numpy_version": np.__version__,
        },
        {"model.json"},
    )


class DeterministicRanker:
    def __init__(
        self,
        provenance: Provenance,
        artifact_id: str,
        observe: Callable[[PredictionInput], None] | None = None,
    ) -> None:
        self.provenance, self.artifact_id, self.observe = provenance, artifact_id, observe
        self.model: FixturePredictor = ConstantModel()
        self.validate(provenance)

    @classmethod
    def load(
        cls,
        directory: Path,
        expected: Provenance,
        observe: Callable[[PredictionInput], None] | None = None,
    ) -> "DeterministicRanker":
        metadata = validate(directory, "atlasrank.deterministic_ranker_fixture", {"model.json"})
        model = json.loads((directory / "model.json").read_text())
        if (
            metadata["feature_schema"] != SCHEMA
            or metadata["feature_schema_id"] != SCHEMA_ID
            or metadata["provenance"] != asdict(expected)
            or metadata["model_role"] != "deterministic_fixture"
            or metadata["numpy_version"] != np.__version__
            or model
            != {
                "role": "deterministic_fixture",
                "trained": False,
                "coefficients": [0.0] * 5,
                "bias": 0.5,
                "feature_schema_id": SCHEMA_ID,
            }
        ):
            raise ValueError("Incompatible fixture schema/model/encoder bundle")
        return cls(expected, metadata["artifact_id"], observe)

    def validate(self, provenance: Provenance) -> None:
        if self.provenance != provenance or provenance.schema_id != SCHEMA_ID or self.model.trained:
            raise ValueError("Incompatible model or feature provenance")

    def predict(self, batch: FeatureBatch) -> NDArray[np.float32]:
        self.validate(batch.provenance)
        actual = final_input(batch)
        if self.observe is not None:
            self.observe(actual)
        prediction = self.model.predict(actual)
        if (
            prediction.dtype != np.float32
            or prediction.shape != (len(batch.keys),)
            or not np.isfinite(prediction).all()
        ):
            raise ValueError("Invalid fixture predictions")
        return prediction
