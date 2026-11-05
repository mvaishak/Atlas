"""Build a source-only DSMLP update from explicit, verified recipe metadata."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

SCRIPTS = ("m7_dsmlp_parity.py", "m7_dsmlp_preflight.py", "run_m7_dsmlp.sh")
MAX_FILE_BYTES = 2 * 1024**2
MAX_ARCHIVE_BYTES = 8 * 1024**2


def read_input(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Expected a small regular input file: {path}")
    return path.read_bytes()


def build_archive(
    repo: Path,
    output: Path,
    *,
    reference_query_recipe: Path,
    reference_provenance: Path,
    descriptor: Path,
    execution_manifest: Path,
) -> dict[str, Any]:
    """Preserve supplied approval state and exclude all external model artifacts."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve existing transfer archives")
    recipe_bytes = read_input(reference_query_recipe)
    recipe = json.loads(recipe_bytes)
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    recipe_digest = hashlib.sha256(canonical.encode()).hexdigest()
    spec = json.loads(read_input(descriptor / "spec.json" if descriptor.is_dir() else descriptor))
    if recipe_digest != spec["query_recipe_id"]:
        raise ValueError("Reference query recipe differs from sealed pin")
    provenance_bytes = read_input(reference_provenance)
    provenance = json.loads(provenance_bytes)
    if (
        provenance.get("recipe_digest") != recipe_digest
        or provenance.get("recipe_file_sha256") != hashlib.sha256(recipe_bytes).hexdigest()
    ):
        raise ValueError("Reference provenance differs from supplied recipe")
    execution_bytes = read_input(execution_manifest)
    execution = json.loads(execution_bytes)
    if type(execution.get("approved")) is not bool:
        raise ValueError("Execution manifest must preserve an explicit boolean approval state")
    contents = {}
    source_root = repo / "src/atlasrank"
    if not (source_root / "__init__.py").is_file():
        raise ValueError("AtlasRank source tree is missing")
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(repo)
        if path.is_symlink():
            raise ValueError(f"Source symlink is unsafe: {relative}")
        if not path.is_file() or "__pycache__" in relative.parts:
            continue
        if (
            path.suffix == ".py"
            or path.name == "py.typed"
            or ("static" in relative.parts and path.suffix in {".html", ".css"})
        ):
            contents[relative.as_posix()] = read_input(path)
    for name in SCRIPTS:
        contents["scripts/" + name] = read_input(repo / "scripts" / name)
    contents["requirements-dsmlp-cpu-v1.txt"] = read_input(
        repo / "environments/dsmlp-cpu/requirements.txt"
    )
    contents["reconstructed-m6-query-recipe-v1.json"] = recipe_bytes
    contents["reconstructed-m6-query-recipe-v1.provenance.json"] = provenance_bytes
    contents["approved-execution.json"] = execution_bytes
    manifest = {
        "schema_version": 1,
        "purpose": "M7 DSMLP source and explicit recipe metadata transfer",
        "query_recipe_id": recipe_digest,
        "files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(contents.items())
        ],
    }
    contents["update-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    if sum(map(len, contents.values())) > MAX_ARCHIVE_BYTES:
        raise ValueError("Source transfer exceeds 8 MiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted(contents.items()):
                    member = tarfile.TarInfo("update-v2/" + name)
                    member.size = len(data)
                    member.mode = 0o700 if name.endswith(".sh") else 0o600
                    archive.addfile(member, io.BytesIO(data))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-query-recipe", type=Path, required=True)
    parser.add_argument("--reference-provenance", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_archive(
        Path(__file__).resolve().parents[1],
        args.output,
        reference_query_recipe=args.reference_query_recipe,
        reference_provenance=args.reference_provenance,
        descriptor=args.descriptor,
        execution_manifest=args.execution_manifest,
    )
    print(json.dumps({"output": str(args.output), "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
