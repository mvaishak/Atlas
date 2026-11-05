from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from atlasrank.training.configuration import parse_manifest
from atlasrank.training.observations import gpu_memory, require_gpu_memory


def proposal() -> dict[str, Any]:
    value: dict[str, Any] = json.loads(Path("configs/m5-colab-pilot.proposed.json").read_text())
    return value


def test_exact_reviewed_proposal_is_accepted() -> None:
    parsed = parse_manifest(proposal())
    assert parsed["seed"] == 42
    assert parsed["batch_size"] == 16
    assert parsed["model_revision"].startswith("1110a243")
    assert parsed["gpu_allocation_seconds"] == 3600


def test_missing_nested_and_conflicting_alias_are_rejected() -> None:
    m = proposal()
    del m["sampling_and_loss"]["temperature"]
    with pytest.raises(ValueError, match="temperature"):
        parse_manifest(m)
    m = proposal()
    m["temperature"] = 0.1
    with pytest.raises(ValueError, match="Conflicting"):
        parse_manifest(m)
    m = copy.deepcopy(proposal())
    m["sampling_and_loss"]["learning_rate"] = 0.01
    with pytest.raises(ValueError, match="learning_rate"):
        parse_manifest(m)


def test_free_vram_is_distinct_from_total_allocated_reserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (7, 24))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 6)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4)
    observed = gpu_memory()
    assert observed == {
        "cuda_free_bytes": 7,
        "cuda_total_bytes": 24,
        "cuda_allocated_bytes": 3,
        "cuda_reserved_bytes": 6,
        "cuda_peak_allocated_bytes": 4,
    }
    with pytest.raises(RuntimeError, match="free VRAM"):
        require_gpu_memory(minimum_free=8, maximum_allocated=10)
    require_gpu_memory(minimum_free=7, maximum_allocated=4)
    with pytest.raises(RuntimeError, match="allocated VRAM"):
        require_gpu_memory(minimum_free=7, maximum_allocated=3)


def test_cuda_rng_restore_uses_cpu_tensors_and_checks_device_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import random

    import numpy as np

    from atlasrank.training.checkpoints import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    states = [torch.get_rng_state().clone(), torch.get_rng_state().clone()]
    states[1][0] ^= 1  # Distinct device states detect reordering or dropping a device.
    restored: list[torch.Tensor] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: states)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: restored.extend(values))
    expected_cpu = torch.get_rng_state().clone()
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    save_checkpoint(
        tmp_path,
        global_step=0,
        sampler_cursor=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        provenance={"fixture": 1},
    )
    torch.rand(4)
    random.random()
    np.random.random()
    real_load = torch.load

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["map_location"] == "cpu"
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    load_checkpoint(
        tmp_path / "step-000000.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        expected_provenance={"fixture": 1},
        map_location="cuda:0",
    )
    assert torch.equal(torch.get_rng_state(), expected_cpu)
    assert random.getstate() == expected_python
    assert np.array_equal(np.random.get_state()[1], expected_numpy[1])
    assert len(restored) == 2
    for actual, expected in zip(restored, states, strict=True):
        assert actual.device.type == "cpu"
        assert torch.equal(actual, expected)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA RNG"):
        load_checkpoint(
            tmp_path / "step-000000.pt",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            expected_provenance={"fixture": 1},
        )


def test_progress_has_finite_gradient_norm_and_host_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlasrank.training.loop import train_batches
    from atlasrank.training.observations import progress_observation

    monkeypatch.setattr(
        "atlasrank.training.observations.gpu_memory",
        lambda: {
            "cuda_free_bytes": 8,
            "cuda_total_bytes": 24,
            "cuda_allocated_bytes": 3,
            "cuda_reserved_bytes": 5,
            "cuda_peak_allocated_bytes": 4,
        },
    )
    model = torch.nn.Linear(1, 1)
    before = model.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    events: list[dict[str, int | float]] = []
    train_batches(
        [torch.ones((1, 1))],
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss_for_batch=lambda batch: model(batch).square().mean(),
        max_global_steps=1,
        on_progress=lambda step, cursor, loss, norm: events.append(
            {**progress_observation(norm), "step": step, "cursor": cursor, "loss": loss}
        ),
    )
    assert not torch.equal(before, model.weight)
    assert events[0]["gradient_norm_before_clip"] >= 0
    assert events[0]["host_available_ram_bytes"] > 0
    assert events[0]["worker_rss_bytes"] > 0
    assert events[0]["step"] == events[0]["cursor"] == 1
    json.dumps(events[0], allow_nan=False)
    with pytest.raises(FloatingPointError):
        progress_observation(float("nan"))


def test_nonfinite_gradient_stops_before_weight_update() -> None:
    from atlasrank.training.loop import train_batches

    model = torch.nn.Linear(1, 1)
    before = model.weight.detach().clone()
    model.weight.register_hook(  # type: ignore[no-untyped-call]
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    with pytest.raises(FloatingPointError, match="gradient norm"):
        train_batches(
            [torch.ones((1, 1))],
            model=model,
            optimizer=optimizer,
            scheduler=None,
            loss_for_batch=lambda batch: model(batch).square().mean(),
            max_global_steps=1,
        )
    assert torch.equal(before, model.weight)
    assert not optimizer.state


def test_replacement_is_nested_and_conflicting_schedule_fails() -> None:
    from atlasrank.training.configuration import PATHS

    raw = json.loads(Path("configs/m5-colab-pilot.corrected.proposed.json").read_text())
    assert not set(PATHS).intersection(raw)
    assert parse_manifest(raw)["max_optimizer_steps"] == 574
    raw["sampling_and_loss"]["maximum_optimizer_steps"] = 573
    with pytest.raises(ValueError, match="Conflicting nested"):
        parse_manifest(raw)
    raw = proposal()
    raw["loss"] = "different objective"
    with pytest.raises(ValueError, match="Conflicting"):
        parse_manifest(raw)
