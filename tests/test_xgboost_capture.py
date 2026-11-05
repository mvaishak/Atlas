"""Native-input reconstruction guards without importing XGBoost on macOS."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_ranking_features import runtime

from atlasrank.features.ranking import build_features
from atlasrank.ranking.prediction import final_input


def prepared(tmp_path: Path) -> tuple[Any, Any]:
    from atlasrank.ranking.xgboost_boundary import NativeBatch

    inputs = final_input(build_features(runtime(tmp_path), "apple", [("us", "b"), ("us", "a")]))
    values = inputs.values
    csr = SimpleNamespace(
        dtype=np.dtype("float32"),
        indptr=np.array([0, 5, 10]),
        indices=np.tile(np.arange(5), 2),
        data=values.reshape(-1).copy(),
    )
    matrix: Any = SimpleNamespace(
        get_data=lambda: csr,
        num_row=lambda: 2,
        num_col=lambda: 5,
        feature_names=list(inputs.names),
        feature_types=["float"] * 5,
    )
    return inputs, NativeBatch(matrix, inputs.keys, "float32")


def test_native_capture_preserves_explicit_zero_cells(tmp_path: Path) -> None:
    from atlasrank.ranking.prediction import assert_exact_parity
    from atlasrank.ranking.xgboost_boundary import capture_native

    inputs, native = prepared(tmp_path)
    captured = capture_native(native, inputs)
    assert_exact_parity(inputs, captured)
    assert (captured.values == 0).any() and not captured.missing.any()


@pytest.mark.parametrize("fault", ["names", "values", "dtype", "missing_zero", "identities"])
def test_native_capture_rejects_boundary_changes(tmp_path: Path, fault: str) -> None:
    from atlasrank.ranking.xgboost_boundary import capture_native

    inputs, native = prepared(tmp_path)
    if fault == "names":
        native.matrix.feature_names.reverse()
    elif fault == "values":
        data = native.matrix.get_data().data.reshape(2, 5)
        data[:, [0, 1]] = data[:, [1, 0]]
    elif fault == "dtype":
        native = replace(native, source_dtype="float64")
    elif fault == "identities":
        native = replace(native, keys=tuple(reversed(native.keys)))
    else:
        csr = native.matrix.get_data()
        assert csr.data[5] == 0
        # An absent CSR cell is missing, even when toarray() would display zero.
        csr.data = np.delete(csr.data, 5)
        csr.indices = np.delete(csr.indices, 5)
        csr.indptr[-1] -= 1
    with pytest.raises(ValueError, match="Native"):
        capture_native(native, inputs)


def test_native_adapter_refuses_mac_before_library_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import atlasrank.ranking.xgboost_boundary as boundary

    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux"):
        boundary.ZeroUpdateModel()
