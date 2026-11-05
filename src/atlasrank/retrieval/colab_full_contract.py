"""Standard-library gates for full-catalog execution; proposals never run workloads."""

import json
from pathlib import Path
from typing import Any

from atlasrank.retrieval.colab_environment import reviewed_source_hashes
from atlasrank.retrieval.colab_transport import sha

ROWS = 1215854
CATALOG_SHA = "70b15aa26c285d76d0ffc10a43f04392f514954c401c20fdeb5337bb429ad309"
AUDIT_SHA = "467b743bb925e9263faf0be08c462e9f0be409a31129c33c962297bc745471d8"
GIB = 1024**3


def verify_inputs(inputs: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if sha(inputs / "FULL_TRANSFER.json") != spec["transfer_manifest_sha256"]:
        raise ValueError("Full transfer manifest differs")
    transfer: dict[str, Any] = json.loads((inputs / "FULL_TRANSFER.json").read_text())
    if transfer["kind"] != "m4_full_label_free_inputs" or transfer["rows"] != ROWS:
        raise ValueError("Wrong full-catalog transfer scope")
    from atlasrank.retrieval.colab_environment import ROOT

    cpu = json.loads((ROOT / "configs/dense.json").read_text())
    allowed = {
        "products.parquet",
        "CATALOG.json",
        "ORIGIN.json",
        "REFERENCE_RECIPE.json",
        "source/src/atlasrank/py.typed",
    }
    allowed |= {"model/" + e["name"] for e in cpu["files"]}
    allowed |= {"source/" + name for name in reviewed_source_hashes()}
    if set(transfer["files"]) != allowed:
        raise ValueError("Export contains missing or non-whitelisted data/source files")
    expected = set(transfer["files"]) | {"FULL_TRANSFER.json"}
    actual = set()
    for p in inputs.rglob("*"):
        if p.is_symlink():
            raise ValueError("Input symlink forbidden")
        if p.is_file() and "__pycache__" not in p.parts:
            actual.add(str(p.relative_to(inputs)))
    if actual != expected:
        raise ValueError("Unexpected full input inventory")
    for name, pin in transfer["files"].items():
        p = inputs / name
        if not p.resolve().is_relative_to(inputs.resolve()) or sha(p) != pin:
            raise ValueError("Full input bytes differ")
    if sum((inputs / n).stat().st_size for n in transfer["files"]) > spec["upload_limit_bytes"]:
        raise ValueError("Input transfer exceeds reservation")
    contract = json.loads((inputs / "CATALOG.json").read_text())
    from hashlib import sha256

    reference = json.loads((inputs / "REFERENCE_RECIPE.json").read_text())
    recipe_id = sha256(
        json.dumps(reference, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        recipe_id != spec["cuda_recipe_id"]
        or transfer["model_files_sha256"] != reference["model_files_sha256"]
    ):
        raise ValueError("Accepted model/encoder recipe differs")
    if (
        contract["rows"] != ROWS
        or contract["source_catalog_sha256"] != CATALOG_SHA
        or contract["catalog_sha256"] != spec["canonical_projection_sha256"]
        or contract["chunk_rows"] != 1024
    ):
        raise ValueError("Canonical full catalog contract differs")
    return transfer


def approve_full(
    path: Path, inputs: Path, persistent: Path, *, local: bool = False, verify_payload: bool = True
) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text())
    flag = "local_execution_approved" if local else "execution_approved"
    if spec.get(flag) is not True or spec.get("runnable") is not True:
        raise PermissionError(
            "Full execution requires explicit lead approval of the pinned manifest"
        )
    frozen = {
        "schema_version": 1,
        "kind": "m4_colab_full_catalog",
        "catalog_products": ROWS,
        "catalog_sha256": CATALOG_SHA,
        "prepared_audit_sha256": AUDIT_SHA,
        "batch_size": 32,
        "component_rtol": 1e-4,
        "component_atol": 1e-5,
        "cosine_atol": 1e-4,
        "model_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "pilot_managed_seconds_already_charged": 96.667498707,
        "chunk_rows": 1024,
        "precision": "float32",
        "wall_seconds": 18000,
        "gpu_seconds_cap": 18000,
        "persistent_limit_bytes": 4 * GIB,
        "scratch_limit_bytes": 6 * GIB,
        "rss_limit_bytes": 8 * GIB,
        "available_ram_start_bytes": 4 * GIB,
        "vram_limit_bytes": 4 * GIB,
        "gpu_name": "NVIDIA L4",
        "device": "cuda:0",
        "tf32": False,
        "autocast": False,
        "dependency_download_limit_bytes": 512 * 1024**2,
        "upload_limit_bytes": 512 * 1024**2,
        "local_wall_seconds": 7200,
        "local_new_artifact_limit_bytes": 8 * GIB,
    }
    if any(spec.get(k) != v for k, v in frozen.items()):
        raise ValueError("Unreviewed full execution limits or corpus")
    if spec.get("previous_manifest_sha256") is not None:
        raise ValueError("Full manifest amendments require separate review")
    if spec["source_sha256"] != reviewed_source_hashes():
        raise ValueError("Reviewed full source pins differ")
    prior = spec.get("prior_allocation_charge_seconds")
    if (
        not spec.get("prior_allocation_charge_approved")
        or not isinstance(prior, (int, float))
        or not spec["pilot_managed_seconds_already_charged"] <= prior < 18000
    ):
        raise PermissionError(
            "Prior total GPU allocation charge requires review, including idle time"
        )
    if not spec.get("operator_owner_token") or not spec.get("storage_approved"):
        raise PermissionError("Full destination and exclusive owner require approval")
    if not local:
        target = Path(spec["persistent_destination"])
        prefix = Path("/content/drive/MyDrive")
        if (
            target.resolve() != persistent.resolve()
            or not target.resolve().is_relative_to(prefix)
            or target == prefix
        ):
            raise ValueError("Full private Drive destination differs")
    if verify_payload:
        verify_inputs(inputs, spec)
    return spec
