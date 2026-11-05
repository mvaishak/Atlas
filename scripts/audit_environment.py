"""Record local compatibility evidence without network access or model execution."""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def audit() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    hardware: dict[str, object] = {}
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "-json", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            details = json.loads(result.stdout)["SPHardwareDataType"][0]
            # Don't copy serial numbers or user environment variables.
            hardware = {
                key: details[key]
                for key in ("machine_name", "chip_type", "physical_memory", "number_processors")
                if key in details
            }
    elif sys.platform.startswith("linux"):
        hardware["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    versions: dict[str, str | None] = {}
    for name in (
        "atlasrank",
        "mlflow",
        "pytest",
        "ruff",
        "mypy",
        "setuptools",
        "wheel",
        "numpy",
        "torch",
        "xgboost",
        "faiss-cpu",
        "polars",
        "pyarrow",
        "psutil",
        "bm25s",
        "fastapi",
        "uvicorn",
        "httpx",
        "xgboost-cpu",
        "sentence-transformers",
        "transformers",
        "huggingface-hub",
        "safetensors",
        "tokenizers",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    disk = shutil.disk_usage(root)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "hardware": hardware,
        "disk_bytes": {"total": disk.total, "used": disk.used, "free": disk.free},
        "installed_distributions_in_audit_interpreter": versions,
        "git_directory_present": (root / ".git").exists(),
        "mac_libomp_present": Path("/opt/homebrew/opt/libomp/lib/libomp.dylib").exists(),
        "local_esci_raw_files_present": any((root / "data" / "esci" / "raw").glob("*/*.parquet")),
        "dataset_downloaded_by_this_audit": False,
        "model_executed_by_this_audit": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
