"""Checksummed immutable file transport; no assumption of atomic Drive rename/fsync."""

import hashlib
import json
import shutil
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def once(path: Path, value: object) -> None:
    with path.open("x") as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        f.flush()


def safe_extract(archive: Path | BytesIO, destination: Path, limit: int) -> None:
    """Trusted project ZIPs only; reject traversal, links, duplicates and zip bombs."""
    if destination.exists():
        raise FileExistsError("Import destination must be fresh")
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if (
            len({i.filename for i in infos}) != len(infos)
            or sum(i.file_size for i in infos) > limit
        ):
            raise ValueError("Duplicate/oversized transport archive")
        for i in infos:
            p = Path(i.filename)
            mode = i.external_attr >> 16
            if p.is_absolute() or ".." in p.parts or (mode & 0o170000) == 0o120000:
                raise ValueError("Unsafe transport archive")
        destination.mkdir(parents=True)
        z.extractall(destination)


def extract_inputs(archive: Path, destination: Path, expected_sha256: str) -> None:
    if archive.stat().st_size > 256 * 1024**2 or sha(archive) != expected_sha256:
        raise ValueError("Input ZIP differs from reviewed transfer")
    safe_extract(archive, destination, 256 * 1024**2)


def publish(directory: Path, store: Path, name: str, limit: int) -> dict[str, Any]:
    """A receipt is the commit point; interrupted uploads remain preserved/ignored."""
    store.mkdir(parents=True, exist_ok=True)
    if not name or Path(name).name != name:
        raise ValueError("Invalid transport name")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Transport symlinks forbidden")
    hashes = {str(p.relative_to(directory)): sha(p) for p in files}
    receipt = store / f"{name}.receipt.json"
    if receipt.exists():
        prior: dict[str, Any] = json.loads(receipt.read_text())
        if Path(prior["archive"]).name != prior["archive"]:
            raise ValueError("Unsafe persistent archive name")
        if prior["files"] != hashes or sha(store / prior["archive"]) != prior["sha256"]:
            raise ValueError("Existing persistent object differs/corrupt")
        return prior
    # UUID avoids overwriting even an interrupted previous upload of this name.
    target = store / f"{name}.{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(directory)))
    if target.stat().st_size > limit:
        raise RuntimeError("Transport object exceeds storage budget; preserve it")
    payload = {"schema_version": 1, "archive": target.name, "sha256": sha(target), "files": hashes}
    # Read the persisted bytes and validate the full compressed inventory before receipt.
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None or set(z.namelist()) != set(hashes):
            raise ValueError("Persistent upload verification failed")
        for entry, expected in hashes.items():
            if hashlib.sha256(z.read(entry)).hexdigest() != expected:
                raise ValueError("Persistent upload file differs")
    once(receipt, payload)
    return payload


def restore(store: Path, name: str, destination: Path, limit: int) -> None:
    payload = json.loads((store / f"{name}.receipt.json").read_text())
    if Path(payload["archive"]).name != payload["archive"]:
        raise ValueError("Unsafe persistent archive name")
    archive = store / payload["archive"]
    if archive.stat().st_size > limit or sha(archive) != payload["sha256"]:
        raise ValueError("Persistent archive checksum/size differs")
    safe_extract(archive, destination, limit)
    actual = {
        str(p.relative_to(destination)): sha(p) for p in destination.rglob("*") if p.is_file()
    }
    if actual != payload["files"]:
        raise ValueError("Restored file inventory/checksums differ")


def copy_verified(source: Path, target: Path, expected: str) -> None:
    if sha(source) != expected:
        raise ValueError("Source checksum differs")
    if target.exists():
        if sha(target) != expected:
            raise ValueError("Existing persistent metadata differs")
        return
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha(target) != expected:
        raise ValueError("Persistent metadata verification failed")
