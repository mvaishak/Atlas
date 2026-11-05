from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from atlasrank.training.configuration import parse_manifest, validate_execution_scope


def test_full_run_scope_retains_frozen_training_settings() -> None:
    raw = json.loads(Path("configs/m5-colab-diagnostic.proposed.json").read_text())
    raw["execution_scope"] = {
        "mode": "sanity_then_training",
        "max_completed_updates": 574,
        "max_active_seconds": 2700,
        "max_new_allocation_seconds": 3600,
        "automatic_full_pilot_continuation": True,
    }
    raw["resource_limits"]["prior_m5_allocation_seconds"] = 0
    parsed = parse_manifest(raw)
    assert validate_execution_scope(parsed)["max_completed_updates"] == 574
    raw["sampling_and_loss"]["learning_rate"] = 3e-5
    with pytest.raises(ValueError):
        parse_manifest(raw)


def test_sanity_requires_updated_finite_weights_and_safe_memory() -> None:
    from atlasrank.training.runner import _model_digest, _validate_sanity

    model = torch.nn.Linear(2, 1)
    before = _model_digest(model)
    event = {
        "loss": 1.2,
        "gradient_norm_before_clip": 0.4,
        "cuda_free_bytes": 9 * 1024**3,
        "cuda_allocated_bytes": 2 * 1024**3,
        "cumulative_peak_allocated_bytes": 3 * 1024**3,
    }
    with pytest.raises(RuntimeError, match="weights"):
        _validate_sanity(model, before, event)
    with torch.no_grad():
        model.weight.add_(0.01)
    _validate_sanity(model, before, event)
    for field, value in [
        ("loss", float("nan")),
        ("gradient_norm_before_clip", float("inf")),
        ("cuda_free_bytes", 7 * 1024**3),
        ("cumulative_peak_allocated_bytes", 11 * 1024**3),
    ]:
        with pytest.raises(RuntimeError):
            _validate_sanity(model, before, dict(event, **{field: value}))
