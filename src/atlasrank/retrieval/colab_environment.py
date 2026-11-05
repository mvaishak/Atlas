"""Standard-library Colab bootstrap contracts. Never install or replace native Torch."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from atlasrank.retrieval.colab_transport import sha

ROOT = Path(__file__).resolve().parents[3]


def reviewed_source_hashes() -> dict[str, str]:
    paths = [
        *sorted((ROOT / "src/atlasrank").rglob("*.py")),
        *sorted((ROOT / "environments/colab").iterdir()),
        ROOT / "configs/dense.json",
        ROOT / "configs/dense-cuda.json",
        ROOT / "docs/EVALUATION.md",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def inspect_runtime() -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_file": torch.__file__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        ).stdout,
    }
    if result["cuda_available"]:
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "vram_free_total": list(torch.cuda.mem_get_info(0)),
            }
        )
    return result


def validate_runtime(info: dict[str, Any], environment: dict[str, Any]) -> None:
    if (
        sys.version_info[:2] != (3, 13)
        or platform.system() != "Linux"
        or info["torch_version"].split("+")[0] != environment["native_torch_base_version"]
        or info["torch_version"] != environment["native_torch_full_version"]
        or not info["torch_cuda"]
        or not info["cuda_available"]
        or info.get("gpu_name") != "NVIDIA L4"
        or info.get("device_count") != 1
    ):
        raise RuntimeError("Unreviewed Colab Python/Torch/CUDA/GPU; export discovery for review")


def validate_installed(environment: dict[str, Any]) -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in environment["pinned_packages"]}
    if versions != environment["pinned_packages"]:
        raise RuntimeError("Installed CUDA overlay differs from pinned environment")
    return versions


def approval(path: Path, inputs: Path, persistent: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text())
    if spec.get("execution_approved") is not True:
        raise PermissionError(
            "Pilot requires explicit configuration and persistent-storage approval"
        )
    frozen = {
        "schema_version": 1,
        "kind": "m4_colab_pilot",
        "titles": 4096,
        "gpu_name": "NVIDIA L4",
        "device": "cuda:0",
        "batch_size": 32,
        "chunk_rows": 1024,
        "precision": "float32",
        "tf32": False,
        "autocast": False,
        "wall_seconds": 600,
        "gpu_allocation_seconds": 600,
        "vram_limit_bytes": 4 * 1024**3,
        "rss_limit_bytes": 8 * 1024**3,
        "available_ram_start_bytes": 4 * 1024**3,
        "persistent_limit_bytes": 1024**3,
        "upload_limit_bytes": 256 * 1024**2,
        "return_limit_bytes": 256 * 1024**2,
        "dependency_download_limit_bytes": 512 * 1024**2,
        "scratch_limit_bytes": 3 * 1024**3,
        "component_rtol": 1e-4,
        "component_atol": 1e-5,
        "cosine_atol": 1e-4,
        "warmup_batches": 2,
    }
    if any(spec.get(k) != v for k, v in frozen.items()):
        raise ValueError("Pilot configuration differs from review proposal")
    approved = spec.get("persistent_destination")
    prefix = Path("/content/drive/MyDrive")
    if (
        not approved
        or persistent.resolve() != Path(approved).resolve()
        or not persistent.resolve().is_relative_to(prefix)
        or persistent.resolve() == prefix
        or not spec.get("operator_owner_token")
        or not spec.get("storage_approved")
    ):
        raise PermissionError(
            "Exact private Drive directory and exclusive ownership must be approved"
        )
    if spec["source_sha256"] != reviewed_source_hashes():
        raise ValueError("Reviewed source/environment/configuration differs")
    transfer = json.loads((inputs / "TRANSFER.json").read_text())
    if sha(inputs / "TRANSFER.json") != spec["transfer_manifest_sha256"]:
        raise ValueError("Pilot transfer manifest differs")
    if transfer["kind"] != "m4_cpu_exact_label_free_pilot" or transfer["rows"] != 4096:
        raise ValueError("Only the reviewed CPU pilot sample is admissible")
    for name, pin in transfer["files"].items():
        target = inputs / name
        if not target.resolve().is_relative_to(inputs.resolve()) or sha(target) != pin:
            raise ValueError("Pilot transferred input differs")
    if sum((inputs / n).stat().st_size for n in transfer["files"]) > spec["upload_limit_bytes"]:
        raise ValueError("Pilot transfer budget exceeded")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect"])
    parser.parse_args()
    print(json.dumps(inspect_runtime(), indent=2))


if __name__ == "__main__":
    main()
