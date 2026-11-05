"""Resource observations for launch gates and existing progress evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import psutil
import torch


def gpu_memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(0)
    return {
        "cuda_free_bytes": free,
        "cuda_total_bytes": total,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(0),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(0),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
    }


def require_gpu_memory(*, minimum_free: int, maximum_allocated: int) -> None:
    memory = gpu_memory()
    if memory["cuda_free_bytes"] < minimum_free:
        raise RuntimeError("M5 minimum free VRAM gate failed")
    if memory["cuda_peak_allocated_bytes"] > maximum_allocated:
        raise RuntimeError("M5 peak allocated VRAM cap exceeded")


def progress_observation(gradient_norm: float) -> dict[str, int | float]:
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("Non-finite progress gradient norm")
    return {
        **gpu_memory(),
        "gradient_norm_before_clip": gradient_norm,
        "host_available_ram_bytes": psutil.virtual_memory().available,
        "worker_rss_bytes": psutil.Process().memory_info().rss,
    }


class DurableEvents:
    """Append scalar-only JSON evidence durably before control returns to a guard."""

    def __init__(self, path: Path) -> None:
        self.stream = path.open("x", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        import json
        import os

        encoded = json.dumps(event, sort_keys=True, allow_nan=False)
        self.stream.write(encoded + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()
