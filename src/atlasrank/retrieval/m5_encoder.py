"""Trained M5 inference recipes, explicitly distinct from the frozen M4 encoder.

No Torch import until construction: index/evidence commands remain native-only.
"""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlasrank.artifacts.dense import file_hash
from atlasrank.data.splits import normalize_query


def verify_file(path: Path, expected: str) -> None:
    if path.is_symlink() or file_hash(path) != expected:
        raise ValueError(f"Artifact hash differs: {path}")


def verify_model(directory: Path, files: dict[str, str]) -> None:
    if not files or "model.safetensors" not in files or directory.is_symlink():
        raise ValueError("Missing model inventory")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe model inventory path")
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe model inventory path")
        verify_file(path, expected)
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError("Model inventory differs")


class M5Encoder:
    """Historical catalog dispatch is 1024; model microbatches remain 32.

    Catalog ST.encode sorts the entire dispatch by length and pads each internal
    batch dynamically, then applies a second normalization. Queries/Track A use
    the existing fixed-256 forward path without that extra normalization.
    """

    def __init__(
        self, directory: Path, pins: dict[str, Any], *, device: str, catalog: bool = False
    ) -> None:
        verify_model(directory, pins["model_files"])
        if importlib.metadata.version("sentence-transformers") != "6.1.0":
            raise ValueError("M5 serialization/execution requires Sentence Transformers 6.1.0")
        from atlasrank.training.model import SharedMiniLM

        if device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("Use a reviewed CPU or single CUDA runtime")
        import torch

        self.device_type = device.split(":")[0]
        if self.device_type == "cpu":
            torch.set_num_threads(1)
        self.shared = SharedMiniLM(str(directory), device)
        self.shared.model.eval()
        self.catalog = catalog
        self.recipe: dict[str, Any] = {
            "kind": "m5-manual-catalog-v1" if catalog else "m5-fixed-query-v1",
            "dimensions": 384,
            "batch_size": 1024 if catalog else 32,
            "model_batch_size": 32,
            "max_length": 256,
            "dtype": "float32",
            "preprocessing": "normalize_query:NFKC-casefold-whitespace",
            "padding": "dynamic-length-sorted" if catalog else "fixed-256",
            "extra_normalization": catalog,
            "device": device,
            "model_files": pins["model_files"],
            "base_revision": pins["base_revision"],
            "sentence_transformers": "6.1.0",
            "runtime": {
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "python": platform.python_version(),
                "cuda": torch.version.cuda,
                "threads": torch.get_num_threads(),
                "autocast": False,
                "cuda_matmul_tf32": (
                    torch.backends.cuda.matmul.allow_tf32 if self.device_type == "cuda" else None
                ),
            },
        }

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts or len(texts) > self.recipe["batch_size"]:
            raise ValueError("Nonempty inputs within recipe dispatch size required")
        import torch

        with torch.inference_mode(), torch.autocast(device_type=self.device_type, enabled=False):
            if self.catalog:
                value = self.shared.model.encode(
                    [normalize_query(t) for t in texts],
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            else:
                value = self.shared._forward(texts).cpu().numpy()
        result = np.asarray(value)
        if (
            result.dtype != np.float32
            or result.shape != (len(texts), 384)
            or not np.isfinite(result).all()
            or not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5, rtol=0)
        ):
            raise ValueError("Invalid M5 normalized vectors")
        return result


def verify_recipe(recipe: dict[str, Any], pins: dict[str, Any], *, catalog: bool = False) -> None:
    """Reject query/product recipe substitution before using saved vectors."""
    expected = {
        "kind": "m5-manual-catalog-v1" if catalog else "m5-fixed-query-v1",
        "dimensions": 384,
        "batch_size": 1024 if catalog else 32,
        "model_batch_size": 32,
        "max_length": 256,
        "dtype": "float32",
        "preprocessing": "normalize_query:NFKC-casefold-whitespace",
        "padding": "dynamic-length-sorted" if catalog else "fixed-256",
        "extra_normalization": catalog,
        "model_files": pins["model_files"],
        "base_revision": pins["base_revision"],
        "sentence_transformers": "6.1.0",
    }
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise ValueError("M5 encoder recipe differs")
