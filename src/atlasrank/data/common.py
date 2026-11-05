"""Deterministic records and conservative local resource budgets."""

import hashlib
import json
import resource
import shutil
import sys
from pathlib import Path

GIB = 1024**3
DISK_LIMIT = 30 * GIB
MEMORY_LIMIT = 8 * GIB
FREE_RESERVE = 2 * GIB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def check_disk(root: Path, additional: int = 0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if additional < 0 or tree_bytes(root) + additional > DISK_LIMIT:
        raise RuntimeError("Local data would exceed the approved 30 GiB reservation")
    if shutil.disk_usage(root).free - additional < FREE_RESERVE:
        raise RuntimeError("Insufficient disk headroom: preserve 2 GiB free")


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def check_memory() -> None:
    if peak_rss_bytes() > MEMORY_LIMIT:
        raise RuntimeError("Preprocessing exceeded the approved 8 GiB RSS ceiling")
