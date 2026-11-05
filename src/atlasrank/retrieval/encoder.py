"""Pinned local MiniLM recipe; CPU inference only, without fitting or Hub fallback."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atlasrank.data.common import sha256
from atlasrank.data.splits import normalize_query

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_FILES = frozenset(
    {
        "1_Pooling/config.json",
        "README.md",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
)


def configuration(path: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text())
    frozen = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "dimensions": 384,
        "max_length": 256,
        "batch_size": 32,
        "device": "cpu",
        "dtype": "float32",
        "attention": "eager",
        "threads": 1,
        "seed": 42,
        "preprocessing": "nfkc_casefold_whitespace_v1",
        "pilot_titles": 4096,
        "pilot_encoding_seconds": 600,
        "pilot_available_ram_bytes": 4 * 1024**3,
        "pilot_incremental_disk_bytes": 1024**3,
        "full_catalog_encoding_authorized": False,
    }
    if any(spec.get(k) != v for k, v in frozen.items()):
        raise ValueError("Unsupported or unfrozen CPU calibration configuration")
    entries = spec["files"]
    if len(entries) != len(MODEL_FILES) or {e["name"] for e in entries} != MODEL_FILES:
        raise ValueError("Unexpected pinned model inventory")
    return spec


def verify_snapshot(spec: dict[str, Any]) -> dict[str, str]:
    directory = Path(spec["model_path"])
    hashes = {}
    for entry in spec["files"]:
        path = directory / entry["name"]
        if path.is_symlink() or path.stat().st_size != entry["bytes"]:
            raise ValueError("Local model file differs from pinned size/type")
        if "sha256" in entry:
            if sha256(path) != entry["sha256"]:
                raise ValueError("Local weights checksum differs from upstream pin")
        else:
            data = path.read_bytes()
            git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if git_blob != entry["git_blob"]:
                raise ValueError("Local model metadata differs from upstream Git blob")
        hashes[entry["name"]] = sha256(path)
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(directory).parts
    }
    if actual != MODEL_FILES:
        raise ValueError("Unexpected executable/weight file in local model snapshot")
    return hashes


def validate_vectors(values: NDArray[np.float32], dimensions: int) -> None:
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] != dimensions
        or not np.isfinite(values).all()
        or not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=1e-5)
    ):
        raise ValueError("Expected finite float32 unit vectors with matching dimensions")


class PinnedEncoder:
    def __init__(self, spec: dict[str, Any]) -> None:
        hashes = verify_snapshot(spec)
        self.recipe: dict[str, Any] = {
            "repo": REPO,
            "revision": REVISION,
            "model_files_sha256": hashes,
            "dimensions": 384,
            "batch_size": 32,
            "max_length": 256,
            "device": "cpu",
            "dtype": "float32",
            "attention": "eager",
            "threads": 1,
            "seed": 42,
            "preprocessing": "nfkc_casefold_whitespace_v1",
            "padding": "right_fixed_256",
            "pooling": "attention_mask_mean_then_l2",
            "encoder_code_sha256": sha256(Path(__file__)),
            "runtime_versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "tokenizers",
                    "safetensors",
                    "numpy",
                    "polars",
                    "pyarrow",
                )
            },
            "runtime_platform": {"platform": platform.platform(), "machine": platform.machine()},
        }
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        self.model: Any = SentenceTransformer(
            str(Path(spec["model_path"]).resolve()),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
            model_kwargs={
                "use_safetensors": True,
                "dtype": torch.float32,
                "attn_implementation": "eager",
            },
        )
        self.model.eval()
        if self.model.max_seq_length != 256 or self.model.get_sentence_embedding_dimension() != 384:
            raise ValueError("Pinned model recipe dimensions/truncation differ")
        self.model.tokenizer.padding_side = "right"
        self.model.tokenizer.truncation_side = "right"

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        import torch

        if not texts or len(texts) > 32 or any(not isinstance(t, str) for t in texts):
            raise ValueError("CPU encoding requires 1 to 32 text strings")
        features = self.model.tokenizer(
            [normalize_query(t) for t in texts],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self.model(features)["sentence_embedding"]
        result = np.asarray(output.detach().cpu().numpy(), dtype=np.float32)
        validate_vectors(result, 384)
        return result
