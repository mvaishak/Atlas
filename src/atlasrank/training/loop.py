"""Single-pass training loop with consumed-batch cursor semantics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from atlasrank.training.checkpoints import save_checkpoint


@dataclass(frozen=True)
class TrainProgress:
    global_step: int
    sampler_cursor: int
    losses: tuple[float, ...]
    stopped_at_step_cap: bool
    checkpoint_seconds: float


def train_batches[Batch](
    batches: Sequence[Batch],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_for_batch: Callable[[Batch], Tensor],
    global_step: int = 0,
    sampler_cursor: int = 0,
    max_global_steps: int,
    grad_clip_norm: float = 1.0,
    checkpoint_directory: Path | None = None,
    checkpoint_every: int = 100,
    provenance: dict[str, Any] | None = None,
    guard: Callable[[], None] | None = None,
    microbatches: Callable[[Batch], Sequence[tuple[Batch, int]]] | None = None,
    diagnostic: Callable[[dict[str, Any]], None] | None = None,
    batch_details: Callable[[Batch], dict[str, int]] | None = None,
    on_progress: Callable[[int, int, float, float], None] | None = None,
) -> TrainProgress:
    """Consume each indexed batch once; cursor advances only after a completed update."""
    if not 0 <= sampler_cursor <= len(batches) or global_step < 0:
        raise ValueError("Invalid initial training progress")
    if max_global_steps < global_step or checkpoint_every < 1 or grad_clip_norm <= 0:
        raise ValueError("Invalid training step/checkpoint/gradient limit")
    if checkpoint_directory is not None and not provenance:
        raise ValueError("Persistent checkpoints require full run provenance")
    model.train()
    losses: list[float] = []
    checkpoint_seconds = 0.0
    cursor = sampler_cursor
    step = global_step
    started = time.monotonic()
    phase = "initializing"
    window_step = step + 1
    micro_index = -1
    micro_groups = 0
    details: dict[str, int] = {}
    latest_loss: float | None = None
    latest_norm: float | None = None
    optimizer_applied = False

    def emit(name: str, **extra: Any) -> None:
        nonlocal phase
        phase = name
        if diagnostic is not None:
            elapsed = time.monotonic() - started
            diagnostic(
                {
                    "schema_version": 2,
                    "phase": name,
                    "global_step": step,
                    "sampler_cursor": cursor,
                    "attempted_step": window_step,
                    "microbatch_index": micro_index,
                    "microbatch_groups": micro_groups,
                    **details,
                    "loss": latest_loss,
                    "gradient_norm_before_clip": latest_norm,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": elapsed,
                    "updates_per_second": (step - global_step) / max(elapsed, 1e-9),
                    "optimizer_applied_in_window": optimizer_applied,
                    **extra,
                }
            )

    try:
        while cursor < len(batches) and step < max_global_steps:
            optimizer_applied = False
            latest_loss = latest_norm = None
            micro_index = -1
            micro_groups = 0
            window_step = step + 1
            parts = list(microbatches(batches[cursor])) if microbatches else [(batches[cursor], 1)]
            if not parts or any(count < 1 for _, count in parts):
                raise ValueError("Microbatches require positive query-group counts")
            count_total = sum(count for _, count in parts)
            details = {
                "effective_groups": count_total,
                **(batch_details(batches[cursor]) if batch_details else {}),
            }
            emit("window_start")
            if guard is not None:
                guard()
            optimizer.zero_grad(set_to_none=True)
            effective_loss = 0.0
            for micro_index, (part, micro_groups) in enumerate(parts):  # noqa: B007
                # Index is consumed by the emit closure, not directly in this block.
                details = {
                    "effective_groups": count_total,
                    **(batch_details(part) if batch_details else {}),
                }
                emit("before_forward")
                if guard is not None:
                    guard()
                loss = loss_for_batch(part)
                if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Training loss must be one finite scalar")
                latest_loss = float(loss.detach().cpu())
                emit("after_forward")
                if guard is not None:
                    guard()
                weighted_loss = loss * (micro_groups / count_total)
                weighted_loss.backward()  # type: ignore[no-untyped-call]
                parameters = [p for p in model.parameters() if p.grad is not None]
                if not parameters:
                    raise RuntimeError("Training loss produced no model gradients")
                gradient_norms = torch.stack(
                    [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
                )
                latest_norm = float(gradient_norms.norm())
                del gradient_norms
                if not math.isfinite(latest_norm):
                    raise FloatingPointError("Non-finite gradient norm")
                effective_loss += latest_loss * micro_groups / count_total
                del weighted_loss, loss
                emit("after_backward")
                if guard is not None:
                    guard()
            norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
            latest_norm = float(norm)
            if not math.isfinite(latest_norm):
                raise FloatingPointError("Non-finite gradient norm")
            latest_loss = effective_loss
            emit("before_update")
            optimizer.step()
            optimizer_applied = True
            if scheduler is not None:
                scheduler.step()
            step += 1
            cursor += 1
            losses.append(effective_loss)
            emit("completed_update")
            if on_progress is not None:
                on_progress(step, cursor, effective_loss, latest_norm)
            if checkpoint_directory is not None and step % checkpoint_every == 0:
                checkpoint_started = time.monotonic()
                save_checkpoint(
                    checkpoint_directory,
                    global_step=step,
                    sampler_cursor=cursor,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    provenance=provenance or {},
                )
                checkpoint_seconds += time.monotonic() - checkpoint_started
            if guard is not None:
                guard()
        if checkpoint_directory is not None and (not losses or step % checkpoint_every != 0):
            checkpoint_started = time.monotonic()
            save_checkpoint(
                checkpoint_directory,
                global_step=step,
                sampler_cursor=cursor,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                provenance=provenance or {},
            )
            checkpoint_seconds += time.monotonic() - checkpoint_started
    except BaseException as error:
        failed_phase = phase
        # Nonfinite values must not prevent the terminal event from being serializable.
        if latest_norm is not None and not math.isfinite(latest_norm):
            latest_norm = None
        try:
            emit(
                "terminal_failure",
                failed_phase=failed_phase,
                exception_type=type(error).__name__,
                exception=str(error),
            )
        except Exception:
            pass  # Preserve the original error if the storage system itself failed.
        raise
    return TrainProgress(
        global_step=step,
        sampler_cursor=cursor,
        losses=tuple(losses),
        stopped_at_step_cap=step >= max_global_steps and cursor < len(batches),
        checkpoint_seconds=checkpoint_seconds,
    )
