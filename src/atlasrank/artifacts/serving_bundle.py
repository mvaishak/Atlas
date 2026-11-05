"""Sealed relocation-safe serving descriptors; preflight never imports native ML."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atlasrank.artifacts.dense import file_hash, seal, validate, write_once

Pipeline = Literal["bm25", "dense_m4", "rrf_m4", "m6_xgb_m4"]


class ServingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    components: dict[str, str]
    pins: dict[str, str]
    pipelines: list[Pipeline] = Field(min_length=1, max_length=4)
    query_recipe_id: str | None = None
    minimum_available_bytes: int = Field(default=8 * 1024**3, ge=256 * 1024**2)
    maximum_rss_bytes: int = Field(default=12 * 1024**3, ge=1, le=12 * 1024**3)

    @model_validator(mode="after")
    def compatible(self) -> "ServingSpec":
        required = {"lexical"}
        if any(p != "bm25" for p in self.pipelines) and (
            self.query_recipe_id is None
            or len(self.query_recipe_id) != 64
            or any(c not in "0123456789abcdef" for c in self.query_recipe_id)
        ):
            raise ValueError("Approved CPU query recipe digest required")
        if any(p != "bm25" for p in self.pipelines) and self.minimum_available_bytes < 8 * 1024**3:
            raise ValueError("Neural startup requires at least 8 GiB available memory")
        if any(p != "bm25" for p in self.pipelines):
            required |= {"dense", "model", "dense_config"}
        if "m6_xgb_m4" in self.pipelines:
            required.add("ranker")
        if (
            set(self.components) != required
            or len(set(self.pipelines)) != len(self.pipelines)
            or "bm25" not in self.pipelines
        ):
            raise ValueError(
                "Exact component inventory and unique pipelines including BM25 required"
            )
        for path in self.components.values():
            relative = Path(path)
            if not path or relative.is_absolute() or ".." in relative.parts or path == ".":
                raise ValueError("Components must be contained relative paths")
        if set(self.pins) != set(self.components) or any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in self.pins.values()
        ):
            raise ValueError("Each component must have an external SHA-256 pin")
        return self


def component_path(root: Path, name: str) -> Path:
    path = root / name
    if (
        root.is_symlink()
        or any(p.is_symlink() for p in [path, *path.parents])
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("Symlink or escaping component")
    return path


def pin_file(root: Path, name: str, relative: str) -> Path:
    path = component_path(root, relative)
    if name == "dense_config":
        return path
    # Model inventory is bound to the dense manifest, not an invented model manifest.
    return path / ("config.json" if name == "model" else "manifest.json")


def prepare(
    root: Path,
    output: Path,
    components: dict[str, str],
    pipelines: list[Pipeline],
    query_recipe_id: str | None = None,
) -> None:
    # Validate paths/inventory before touching any caller-selected component.
    minimum = 256 * 1024**2 if pipelines == ["bm25"] else 8 * 1024**3
    candidate = ServingSpec(
        query_recipe_id=query_recipe_id,
        minimum_available_bytes=minimum,
        components=components,
        pipelines=pipelines,
        pins={k: "0" * 64 for k in components},
    )
    pins = {k: file_hash(pin_file(root, k, v)) for k, v in candidate.components.items()}
    spec = ServingSpec(
        components=components,
        pipelines=pipelines,
        pins=pins,
        minimum_available_bytes=minimum,
        query_recipe_id=query_recipe_id,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / "spec.json", spec.model_dump())
    seal(output, {"schema_version": 1, "kind": "atlasrank.serving"}, {"spec.json"})


def preflight(directory: Path, root: Path) -> tuple[ServingSpec, str]:
    metadata = validate(directory, "atlasrank.serving", {"spec.json"})
    spec = ServingSpec.model_validate(json.loads((directory / "spec.json").read_text()))
    for name, relative in spec.components.items():
        try:
            component = component_path(root, relative)
            if component.is_dir() and any(p.is_symlink() for p in component.rglob("*")):
                raise ValueError("Symlink within serving component")
            path = pin_file(root, name, relative)
            if path.is_symlink() or file_hash(path) != spec.pins[name]:
                raise ValueError("Serving component pin differs")
        except OSError as error:
            raise ValueError("Serving component missing or unreadable") from error
    return spec, str(metadata["artifact_id"])
