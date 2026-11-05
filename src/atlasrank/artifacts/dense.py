"""Immutable manifests for chunks/bundles; validate before native FAISS loading."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import polars as pl


class InvalidDense(ValueError):
    """Incomplete, incompatible or corrupt local dense artifacts fail closed."""


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: object) -> None:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def product_keys(products: pl.DataFrame) -> list[tuple[str, str]]:
    if products.columns != ["product_locale", "product_id", "product_title"] or any(
        t != pl.String for t in products.dtypes
    ):
        raise InvalidDense("Invalid product/title schema")
    if not products.height or any(products.null_count().row(0)):
        raise InvalidDense("Expected nonempty products with normalized text")
    keys = list(products.select("product_locale", "product_id").iter_rows())
    if keys != sorted(set(keys)) or any(locale != "us" or not pid.strip() for locale, pid in keys):
        raise InvalidDense("Product keys must be unique, canonical and US")
    return keys


def seal(directory: Path, metadata: dict[str, Any], names: set[str]) -> dict[str, Any]:
    payload = {**metadata, "files": {name: file_hash(directory / name) for name in sorted(names)}}
    manifest = {**payload, "artifact_id": digest(payload)}
    write_once(directory / "manifest.json", manifest)
    write_once(
        directory / "COMPLETE.json",
        {"schema_version": 1, "manifest_sha256": file_hash(directory / "manifest.json")},
    )
    for path in directory.iterdir():
        with path.open("rb") as source:
            os.fsync(source.fileno())
        path.chmod(0o444)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


def validate(directory: Path, kind: str, names: set[str]) -> dict[str, Any]:
    try:
        expected = names | {"manifest.json", "COMPLETE.json"}
        entries = list(directory.rglob("*"))
        if directory.is_symlink() or any(p.is_symlink() for p in entries):
            raise InvalidDense("Symlink dense artifacts aren't supported")
        if {str(p.relative_to(directory)) for p in entries if p.is_file()} != expected:
            raise InvalidDense("Unexpected dense component inventory")
        complete = json.loads((directory / "COMPLETE.json").read_text())
        if complete.get("schema_version") != 1 or complete["manifest_sha256"] != file_hash(
            directory / "manifest.json"
        ):
            raise InvalidDense("Incomplete dense artifact")
        manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != 1 or manifest.get("kind") != kind:
            raise InvalidDense("Unsupported dense artifact schema/kind")
        payload = {k: v for k, v in manifest.items() if k != "artifact_id"}
        if digest(payload) != manifest["artifact_id"] or set(manifest["files"]) != names:
            raise InvalidDense("Dense identity/component manifest differs")
        if any(file_hash(directory / name) != manifest["files"][name] for name in names):
            raise InvalidDense("Dense component checksum differs")
        return manifest
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise InvalidDense("Invalid or unreadable dense artifact") from error
