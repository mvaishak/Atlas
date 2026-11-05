"""Receipt-only full return archives; no native ML dependencies on the cloud side."""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from atlasrank.retrieval.colab_transport import once, safe_extract, sha


def return_archive(store: Path, output: Path, limit: int = 4 * 1024**3) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve previous returns")
    files = []
    selected = set()
    for directory in ("control", "chunks", "evidence", "events"):
        for receipt in sorted((store / directory).glob("*.receipt.json")):
            payload = json.loads(receipt.read_text())
            name = payload["archive"]
            if Path(name).name != name:
                raise ValueError("Unsafe receipt archive")
            archive = receipt.parent / name
            if sha(archive) != payload["sha256"]:
                raise ValueError("Corrupt acknowledged upload")
            files.extend([receipt, archive])
            selected.add(str(archive.relative_to(store)))
    files.extend(sorted(store.glob("*.json")))
    files.extend(sorted((store / "progress").glob("*.json")))
    if not any(p.parent.name == "chunks" for p in files):
        raise ValueError("No persistent chunks")
    if sum(p.stat().st_size for p in files) > limit:
        raise RuntimeError("Return cap exceeded")
    inventory = {str(p.relative_to(store)): sha(p) for p in files}
    unacknowledged = [
        str(p.relative_to(store))
        for p in store.rglob("*.zip")
        if str(p.relative_to(store)) not in selected
    ]
    with zipfile.ZipFile(output, "x", zipfile.ZIP_STORED) as z:
        for p in files:
            if p.is_symlink():
                raise ValueError("Return symlink unsupported")
            z.write(p, str(p.relative_to(store)))
        z.writestr(
            "RETURN.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "files": inventory,
                    "unacknowledged_uploads_preserved_in_drive": unacknowledged,
                },
                sort_keys=True,
            ),
        )
    if output.stat().st_size > limit:
        raise RuntimeError("Archive reservation exceeded")
    report = {
        "archive": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha(output),
        "files": len(inventory),
        "unacknowledged_uploads": unacknowledged,
    }
    once(output.with_suffix(".transfer.json"), report)
    return report


def import_archive(
    archive: Path, expected_sha: str, destination: Path, limit: int = 4 * 1024**3
) -> None:
    if archive.stat().st_size > limit or sha(archive) != expected_sha:
        raise ValueError("Return transfer checksum/size differs")
    safe_extract(archive, destination, limit)
    manifest = json.loads((destination / "RETURN.json").read_text())
    actual = {
        str(p.relative_to(destination)): sha(p)
        for p in destination.rglob("*")
        if p.is_file() and p.name != "RETURN.json"
    }
    if actual != manifest["files"]:
        raise ValueError("Return inventory/checksum differs")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(return_archive(a.store, a.output), indent=2))


if __name__ == "__main__":
    main()


def restore_return_job(archive: Path, job: Path, checkpoint: Any = None) -> None:
    """Stream nested receipt ZIPs without retaining a second compressed catalog copy."""
    import hashlib
    from io import BytesIO

    job.mkdir()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate return entries")
        inventory = json.loads(z.read("RETURN.json"))["files"]
        if set(names) != set(inventory) | {"RETURN.json"}:
            raise ValueError("Return inventory differs")
        # Validate every returned byte, including operational evidence, in bounded streams.
        for name, pin in inventory.items():
            if checkpoint:
                checkpoint()
            h = hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda: f.read(1024**2), b""):
                    h.update(block)
            if h.hexdigest() != pin:
                raise ValueError("Return member checksum differs")
        for folder in ("control", "chunks"):
            for name in sorted(
                n for n in names if n.startswith(folder + "/") and n.endswith(".receipt.json")
            ):
                if checkpoint:
                    checkpoint()
                payload = json.loads(z.read(name))
                member = folder + "/" + payload["archive"]
                if Path(payload["archive"]).name != payload["archive"]:
                    raise ValueError("Unsafe nested archive")
                if z.getinfo(member).file_size > 32 * 1024**2:
                    raise RuntimeError("One checkpoint exceeds bounded restore buffer")
                data = z.read(member)
                if hashlib.sha256(data).hexdigest() != payload["sha256"]:
                    raise ValueError("Nested receipt checksum differs")
                if folder == "chunks":
                    offset = Path(name).name.removesuffix(".receipt.json")
                    if len(offset) != 12 or not offset.isdigit():
                        raise ValueError("Invalid chunk name")
                    target = job / "chunks" / offset
                    safe_extract(BytesIO(data), target, 32 * 1024**2)
                    actual = {
                        str(p.relative_to(target)): sha(p) for p in target.rglob("*") if p.is_file()
                    }
                    if actual != payload["files"]:
                        raise ValueError("Nested chunk inventory differs")
                else:
                    with zipfile.ZipFile(BytesIO(data)) as control:
                        if set(control.namelist()) != set(payload["files"]):
                            raise ValueError("Nested control inventory differs")
                        for rel, pin in payload["files"].items():
                            p = Path(rel)
                            if rel not in {"JOB.json", "JOB_COMPLETE.json"} and not (
                                len(p.parts) == 2
                                and p.parts[0] == "attempts"
                                and p.suffix == ".json"
                            ):
                                raise ValueError("Unexpected nested control path")
                            raw = control.read(rel)
                            if hashlib.sha256(raw).hexdigest() != pin:
                                raise ValueError("Control hash differs")
                            dest = job / rel
                            dest.parent.mkdir(exist_ok=True)
                            with dest.open("xb") as f:
                                f.write(raw)
