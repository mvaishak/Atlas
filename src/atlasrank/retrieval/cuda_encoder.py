"""Separate CUDA execution of the pinned CPU model, tokenizer and pooling modules."""

import hashlib
import importlib.metadata
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlasrank.artifacts.dense import file_hash
from atlasrank.data.splits import normalize_query
from atlasrank.retrieval.encoder import PinnedEncoder, validate_vectors


def token_arrays(tokenizer: Any, texts: list[str]) -> dict[str, NDArray[np.int64]]:
    features = tokenizer(
        [normalize_query(t) for t in texts],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="np",
    )
    return {k: np.asarray(v, dtype=np.int64) for k, v in features.items()}


def token_hashes(features: dict[str, NDArray[np.int64]]) -> list[str]:
    """Row hashes are independent of batching and NPZ serialization metadata."""
    if not features or len({v.shape for v in features.values()}) != 1:
        raise ValueError("Inconsistent tokenizer tensors")
    shape = next(iter(features.values())).shape
    if len(shape) != 2 or shape[1] != 256:
        raise ValueError("Expected fixed 256-token inputs")
    return [
        hashlib.sha256(
            b"".join(
                name.encode() + b"\0" + features[name][i].astype("<i8").tobytes()
                for name in sorted(features)
            )
        ).hexdigest()
        for i in range(shape[0])
    ]


def gpu_report(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pilot requires exactly one available CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    return {
        "name": props.name,
        "total_vram_bytes": total,
        "free_vram_bytes": free,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver_version": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
    }


class CudaEncoder(PinnedEncoder):
    def __init__(self, cpu_spec: dict[str, Any], cuda_spec: dict[str, Any]) -> None:
        # This must be set in the fresh worker before initializing CUDA/cuBLAS.
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("Fresh worker requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        import torch

        hardware = gpu_report(torch)
        if hardware["name"] != cuda_spec["gpu_name"]:
            raise RuntimeError("Allocated GPU differs from approved pilot")
        cap = cuda_spec["vram_limit_bytes"]
        if hardware["free_vram_bytes"] < cap + 1024**3:
            raise RuntimeError("Insufficient free VRAM plus 1 GiB reserve")
        torch.cuda.set_per_process_memory_fraction(cap / hardware["total_vram_bytes"], 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        super().__init__(cpu_spec)  # Existing local-only architecture/weights/pooling loader.
        self.model.to("cuda:0")
        self.model.float()
        self.recipe = {
            **self.recipe,
            "device": "cuda:0",
            "tf32": False,
            "autocast": False,
            "cuda_adapter_sha256": file_hash(Path(__file__)),
            "runtime_platform": {"system": "Linux", "machine": "x86_64"},
            "cuda_runtime": {k: v for k, v in hardware.items() if k != "free_vram_bytes"},
            "native_cuda_package_versions": {
                name: importlib.metadata.version(name)
                for name in sorted({d.metadata["Name"] for d in importlib.metadata.distributions()})
                if name.lower().startswith(("nvidia-", "cuda-", "triton"))
            },
        }
        self.vram_limit = cap
        self.timings: list[dict[str, Any]] = []
        self.last_token_hashes: list[str] = []
        self.check_memory()

    def check_memory(self) -> dict[str, int]:
        import torch

        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        peak_allocated = torch.cuda.max_memory_allocated(0)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        free, _ = torch.cuda.mem_get_info(0)
        if (
            max(allocated, reserved, peak_allocated, peak_reserved) > self.vram_limit
            or free < 1024**3
        ):
            raise RuntimeError("CUDA memory limit/reserve exceeded")
        return {
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "device_free_bytes": free,
        }

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        import torch

        if not texts or len(texts) > 32 or any(not isinstance(t, str) for t in texts):
            raise ValueError("CUDA reference encoding requires 1 to 32 strings")
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        arrays = token_arrays(self.model.tokenizer, texts)
        self.last_token_hashes = token_hashes(arrays)
        tokenized = time.perf_counter()
        features = {name: torch.from_numpy(value).to("cuda:0") for name, value in arrays.items()}
        torch.cuda.synchronize(0)
        transferred = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            output = self.model(features)["sentence_embedding"]
        torch.cuda.synchronize(0)
        inferred = time.perf_counter()
        result = np.asarray(output.detach().cpu().numpy(), dtype=np.float32)
        torch.cuda.synchronize(0)
        finished = time.perf_counter()
        validate_vectors(result, 384)
        self.timings.append(
            {
                "titles": len(texts),
                "tokenization_seconds": tokenized - started,
                "host_to_device_seconds": transferred - tokenized,
                "model_inference_seconds": inferred - transferred,
                "device_to_host_seconds": finished - inferred,
                "end_to_end_seconds": finished - started,
            }
        )
        self.check_memory()
        return result
