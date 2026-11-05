"""Linux-only zero-update validation adapter; never a learned ranker loader.

The existing DeterministicRanker supplies this predictor with its final validated
PredictionInput. Capture reconstructs explicit native missingness at Booster.predict.
Importing this module does not import XGBoost or Torch.
"""

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from atlasrank.ranking.prediction import PredictionInput, assert_exact_parity

if TYPE_CHECKING:
    import xgboost as xgb


@dataclass(frozen=True)
class NativeBatch:
    matrix: "xgb.DMatrix"
    keys: tuple[tuple[str, str], ...]
    source_dtype: str


def capture_native(native: NativeBatch, expected: PredictionInput) -> PredictionInput:
    matrix = native.matrix
    csr = matrix.get_data()
    if (
        native.source_dtype != "float32"
        or csr.dtype != np.float32
        or native.keys != expected.keys
        or (matrix.num_row(), matrix.num_col()) != expected.values.shape
        or tuple(matrix.feature_names or []) != expected.names
        or tuple(matrix.feature_types or []) != ("float",) * 5
    ):
        raise ValueError("Native DMatrix schema/dtype/identity differs")
    # Do not use toarray(): absent cells must remain missing, not become zeros.
    values = np.full(expected.values.shape, np.nan, dtype=np.float32)
    for row in range(matrix.num_row()):
        start, stop = csr.indptr[row : row + 2]
        values[row, csr.indices[start:stop]] = csr.data[start:stop]
    missing = np.isnan(values)
    if not np.array_equal(missing, expected.missing) or not np.array_equal(values, expected.values):
        raise ValueError("Native DMatrix values/missing semantics differ")
    captured = PredictionInput(
        tuple(matrix.feature_names or []),
        (str(csr.dtype),) * 5,
        native.keys,
        values,
        missing,
        expected.text_empty,
        expected.query_fingerprint,
        expected.provenance,
        expected.schema_id,
    )
    assert_exact_parity(expected, captured)
    return captured


class ZeroUpdateModel:
    """Explicit validation fixture injected into the existing prediction adapter."""

    trained = False

    def __init__(self) -> None:
        if sys.platform != "linux":
            raise RuntimeError("Native XGBoost boundary validation is Linux-only")
        import xgboost as xgb

        if xgb.__version__ != "3.1.1":
            raise RuntimeError("Requires the reviewed XGBoost 3.1.1")
        self.xgb = xgb
        self.booster: xgb.Booster | None = None
        self.observations: list[PredictionInput] = []

    def convert(
        self,
        inputs: PredictionInput,
        *,
        values: NDArray[Any] | None = None,
        names: tuple[str, ...] | None = None,
        keys: tuple[tuple[str, str], ...] | None = None,
        missing: float = float("nan"),
    ) -> NativeBatch:
        array = inputs.values if values is None else values
        matrix = self.xgb.DMatrix(
            array,
            feature_names=list(inputs.names if names is None else names),
            feature_types=["float"] * 5,
            missing=missing,
            nthread=1,
        )
        return NativeBatch(matrix, inputs.keys if keys is None else keys, str(array.dtype))

    def predict(self, inputs: PredictionInput) -> NDArray[np.float32]:
        native = self.convert(inputs)
        if self.booster is None:
            self.booster = self.xgb.Booster(
                params={
                    "objective": "rank:ndcg",
                    "ndcg_exp_gain": False,
                    "base_score": 0.5,
                    "device": "cpu",
                    "nthread": 1,
                },
                cache=[native.matrix],
            )
        if self.booster.num_boosted_rounds() != 0:
            raise ValueError("Only a zero-update fixture Booster is permitted")
        # Final boundary: capture this exact DMatrix immediately before native predict.
        captured = capture_native(native, inputs)
        self.observations.append(captured)
        prediction = self.booster.predict(native.matrix, validate_features=True, strict_shape=True)
        return prediction.reshape(-1)
