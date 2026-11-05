"""Source-only DSMLP transfer packaging requires explicit, pinned metadata."""

import hashlib
import json
import runpy
import tarfile
from pathlib import Path
from typing import Any

import pytest


def metadata(tmp_path: Path) -> dict[str, Path]:
    recipe = {"runtime_platform": {"platform": "Linux-fixture", "machine": "x86_64"}}
    encoded = (json.dumps(recipe) + "\n").encode()
    pin = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    values = {
        "reference_query_recipe": encoded,
        "reference_provenance": json.dumps(
            {
                "recipe_digest": pin,
                "recipe_file_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        ).encode(),
        "descriptor": json.dumps({"query_recipe_id": pin}).encode(),
        "execution_manifest": Path("configs/m7-dsmlp-smoke.proposed.json").read_bytes(),
    }
    paths = {}
    for name, data in values.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_bytes(data)
    return paths


def build(tmp_path: Path, **overrides: Any) -> Path:
    inputs = metadata(tmp_path)
    inputs.update(overrides)
    output = tmp_path / "source.tar.gz"
    runpy.run_path("scripts/package_m7_dsmlp.py")["build_archive"](Path.cwd(), output, **inputs)
    return output


def test_source_archive_contains_only_verified_inputs_and_safe_source(tmp_path: Path) -> None:
    output = build(tmp_path)
    with tarfile.open(output) as archive:
        contents = {}
        for member in archive.getmembers():
            assert member.isfile()
            assert member.name.startswith("update-v2/")
            assert not any(
                part in {".local", "__pycache__", "artifacts", "data"}
                for part in Path(member.name).parts[:2]
            )
            stream = archive.extractfile(member)
            assert stream is not None
            contents[member.name.removeprefix("update-v2/")] = stream.read()
    manifest = json.loads(contents.pop("update-manifest.json"))
    assert {record["path"] for record in manifest["files"]} == set(contents)
    for record in manifest["files"]:
        content = contents[record["path"]]
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        assert len(content) == record["bytes"]
    assert "src/atlasrank/artifacts/query_recipe.py" in contents
    assert "scripts/run_m7_dsmlp.sh" in contents
    assert "scripts/m7_dsmlp_parity.py" in contents
    assert "scripts/m7_dsmlp_preflight.py" in contents
    assert (
        contents["requirements-dsmlp-cpu-v1.txt"]
        == Path("environments/dsmlp-cpu/requirements.txt").read_bytes()
    )
    assert not json.loads(contents["approved-execution.json"])["approved"]
    assert "descriptor.json" not in contents
    assert "reconstructed-m6-query-recipe-v1.provenance.json" in contents
    with pytest.raises(FileExistsError):
        build(tmp_path)


@pytest.mark.parametrize("field", ["recipe_digest", "recipe_file_sha256"])
def test_package_rejects_recipe_provenance_mismatch(tmp_path: Path, field: str) -> None:
    inputs = metadata(tmp_path)
    provenance = json.loads(inputs["reference_provenance"].read_text())
    provenance[field] = "0" * 64
    inputs["reference_provenance"].write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance"):
        runpy.run_path("scripts/package_m7_dsmlp.py")["build_archive"](
            Path.cwd(), tmp_path / "out.tar.gz", **inputs
        )
    assert not (tmp_path / "out.tar.gz").exists()


def test_package_rejects_sealed_recipe_mismatch(tmp_path: Path) -> None:
    inputs = metadata(tmp_path)
    inputs["descriptor"].write_text(json.dumps({"query_recipe_id": "0" * 64}))
    with pytest.raises(ValueError, match="sealed"):
        runpy.run_path("scripts/package_m7_dsmlp.py")["build_archive"](
            Path.cwd(), tmp_path / "out.tar.gz", **inputs
        )


def test_runner_requires_explicit_platform_override() -> None:
    runner = Path("scripts/run_m7_dsmlp.sh").read_text()
    assert "PORTABILITY_ARGS=()" in runner
    assert 'case "${1:-}" in' in runner
    assert '    "${PORTABILITY_ARGS[@]}"' in runner


def test_source_archive_is_reproducible(tmp_path: Path) -> None:
    inputs = metadata(tmp_path)
    package = runpy.run_path("scripts/package_m7_dsmlp.py")["build_archive"]
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    package(Path.cwd(), first, **inputs)
    package(Path.cwd(), second, **inputs)
    assert first.read_bytes() == second.read_bytes()


def test_package_works_with_source_only_checkout(tmp_path: Path) -> None:
    inputs = metadata(tmp_path)
    repo = tmp_path / "checkout"
    files = (
        "src/atlasrank/__init__.py",
        "src/atlasrank/artifacts/query_recipe.py",
        "scripts/m7_dsmlp_parity.py",
        "scripts/m7_dsmlp_preflight.py",
        "scripts/run_m7_dsmlp.sh",
        "environments/dsmlp-cpu/requirements.txt",
    )
    for name in files:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    output = tmp_path / "clean-source.tar.gz"
    runpy.run_path("scripts/package_m7_dsmlp.py")["build_archive"](repo, output, **inputs)
    assert output.is_file()
    assert not (repo / ".local").exists()


def test_package_rejects_source_symlink(tmp_path: Path) -> None:
    inputs = metadata(tmp_path)
    repo = tmp_path / "checkout"
    source = repo / "src/atlasrank"
    source.mkdir(parents=True)
    (source / "__init__.py").symlink_to(Path("src/atlasrank/__init__.py").resolve())
    with pytest.raises(ValueError, match="symlink"):
        runpy.run_path("scripts/package_m7_dsmlp.py")["build_archive"](
            repo, tmp_path / "unsafe.tar.gz", **inputs
        )
    assert not (tmp_path / "unsafe.tar.gz").exists()


def test_bootstrap_rejects_unapproved_manifest_before_compute_or_install(tmp_path: Path) -> None:
    import subprocess
    import sys

    update = tmp_path / "update-v2"
    (update / "scripts").mkdir(parents=True)
    for name in ("m7_dsmlp_parity.py", "m7_dsmlp_preflight.py"):
        (update / "scripts" / name).write_bytes((Path("scripts") / name).read_bytes())
    (update / "approved-execution.json").write_bytes(
        Path("configs/m7-dsmlp-smoke.proposed.json").read_bytes()
    )
    runner = Path("scripts/run_m7_dsmlp.sh").read_text()
    precheck = runner.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path), str(tmp_path), str(update)],
        input=precheck,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Approval and the exact bounded M7 CPU execution contract are required" in result.stderr
