"""Validate a lexical bundle completely before constructing its search adapter."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from atlasrank.data.common import sha256, write_json
from atlasrank.retrieval.analyzer import ANALYZER

CORE_FILES = frozenset(
    {
        "products.parquet",
        "backend/data.csc.index.npy",
        "backend/indices.csc.index.npy",
        "backend/indptr.csc.index.npy",
        "backend/vocab.index.json",
        "backend/params.index.json",
    }
)


class InvalidIndex(ValueError):
    """An incompatible, incomplete, or corrupt lexical bundle must stay unready."""


def identity(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def seal(directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    payload = {**metadata, "files": {name: sha256(directory / name) for name in sorted(CORE_FILES)}}
    manifest = {**payload, "bundle_id": identity(payload)}
    write_json(directory / "manifest.json", manifest)
    write_json(
        directory / "COMPLETE.json",
        {
            "schema_version": 1,
            "manifest_sha256": sha256(directory / "manifest.json"),
        },
    )
    return manifest


def validate(directory: Path) -> dict[str, Any]:
    try:
        complete = json.loads((directory / "COMPLETE.json").read_text())
        if not isinstance(complete, dict):
            raise InvalidIndex("Invalid completion record")
        if complete.get("schema_version") != 1:
            raise InvalidIndex("Unsupported completion schema")
        if sha256(directory / "manifest.json") != complete["manifest_sha256"]:
            raise InvalidIndex("Manifest checksum mismatch")
        manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
        if not isinstance(manifest, dict):
            raise InvalidIndex("Invalid lexical manifest")
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "atlasrank.bm25":
            raise InvalidIndex("Unsupported lexical artifact schema")
        payload = {k: v for k, v in manifest.items() if k != "bundle_id"}
        if identity(payload) != manifest.get("bundle_id"):
            raise InvalidIndex("Bundle identity mismatch")
        if manifest.get("bm25s_version") != importlib.metadata.version("bm25s"):
            raise InvalidIndex("BM25 library version mismatch")
        for name in ("numpy", "scipy"):
            if manifest.get(f"{name}_version") != importlib.metadata.version(name):
                raise InvalidIndex("Numerical dependency version mismatch")
        expected = {
            "analyzer": ANALYZER,
            "fields": ["product_title"],
            "method": "lucene",
            "dtype": "float32",
            "backend": "numpy",
            "k1": 1.5,
            "b": 0.75,
        }
        if any(manifest.get(k) != value for k, value in expected.items()):
            raise InvalidIndex("Incompatible analyzer or scoring configuration")
        if not isinstance(manifest.get("num_docs"), int) or manifest["num_docs"] <= 0:
            raise InvalidIndex("Invalid catalog size")
        if set(manifest["files"]) != CORE_FILES:
            raise InvalidIndex("Unexpected lexical component manifest")
        actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
        if actual != CORE_FILES | {"manifest.json", "COMPLETE.json"}:
            raise InvalidIndex("Missing or unexpected lexical component")
        for name in CORE_FILES:
            path = directory / name
            if path.is_symlink() or sha256(path) != manifest["files"][name]:
                raise InvalidIndex("Lexical component checksum mismatch")
        return manifest
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidIndex("Unreadable or incomplete lexical bundle") from error
