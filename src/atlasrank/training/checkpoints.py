"""Immutable, checksummed training checkpoints with exact cursor/RNG recovery."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG state is incompatible with this runtime")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def save_checkpoint(
    directory: Path,
    *,
    global_step: int,
    sampler_cursor: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if global_step < 0 or sampler_cursor < 0 or not provenance:
        raise ValueError("Checkpoint requires nonnegative progress and provenance")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"step-{global_step:06d}.pt"
    receipt_path = destination.with_suffix(".json")
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("Checkpoints are immutable; this step already exists")
    state = {
        "schema_version": 1,
        "global_step": global_step,
        "sampler_cursor": sampler_cursor,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": _rng_state(),
        "provenance": provenance,
    }
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix="checkpoint-tmp-", suffix=".pt", delete=False
    ) as temp:
        temporary = Path(temp.name)
    try:
        torch.save(state, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema_version": 1,
        "file": destination.name,
        "sha256": _digest(destination),
        "bytes": destination.stat().st_size,
        "global_step": global_step,
        "sampler_cursor": sampler_cursor,
        "provenance": provenance,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    with receipt_path.open("rb") as stream:
        os.fsync(stream.fileno())
    return receipt


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_provenance: dict[str, Any],
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    receipt_path = path.with_suffix(".json")
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("Checkpoint or its receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("file") != path.name or receipt.get("sha256") != _digest(path):
        raise ValueError("Checkpoint receipt integrity check failed")
    # Stage on CPU regardless of requested device: model/optimizer loaders copy to
    # their existing parameter devices, while RNG byte tensors must remain on CPU.
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1 or state.get("provenance") != expected_provenance:
        raise ValueError("Checkpoint provenance/schema is incompatible")
    if receipt.get("provenance") != expected_provenance:
        raise ValueError("Checkpoint receipt provenance differs")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        if state["scheduler"] is None:
            raise ValueError("Checkpoint is missing scheduler state")
        scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng"])
    return int(state["global_step"]), int(state["sampler_cursor"])
