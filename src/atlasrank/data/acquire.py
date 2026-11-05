"""Fetch only the pinned official files; never replace an existing raw file."""

import json
import re
import urllib.request
import uuid
from pathlib import Path

from atlasrank.data.common import check_disk, sha256, write_json


def acquire(config: Path, root: Path) -> Path:
    spec = json.loads(config.read_text())
    revision: str = spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable Git revision is required")
    if spec["repository"] != "https://github.com/amazon-science/esci-data":
        raise ValueError("Only the approved official ESCI source is allowed")
    target = root / "raw" / revision
    target.mkdir(parents=True, exist_ok=True)
    needed = sum(f["bytes"] for f in spec["files"] if not (target / f["name"]).exists())
    check_disk(root, needed)
    for entry in spec["files"]:
        name = entry["name"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid local raw filename")
        path = target / name
        if path.exists():
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Existing raw file differs from pin; preserved: {name}")
            continue
        host = {"media": "media.githubusercontent.com/media", "raw": "raw.githubusercontent.com"}[
            entry["transport"]
        ]
        url = f"https://{host}/amazon-science/esci-data/{revision}/{entry['path']}"
        partial = target / f"{name}.{uuid.uuid4().hex}.partial"
        request = urllib.request.Request(url, headers={"User-Agent": "AtlasRank-M2/0.1"})
        received = 0
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as out:
            advertised = response.headers.get("Content-Length")
            if advertised is not None and int(advertised) != entry["bytes"]:
                raise ValueError(f"Source size differs from pin: {name}")
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > entry["bytes"]:
                    raise ValueError(f"Download exceeds pinned size: {name}")
                out.write(chunk)
        if received != entry["bytes"] or sha256(partial) != entry["sha256"]:
            raise ValueError(f"Raw checksum/size mismatch; partial preserved: {name}")
        partial.rename(path)
        path.chmod(0o444)
        print(json.dumps({"event": "raw_verified", "file": name, "bytes": received}), flush=True)
    provenance = target / "provenance.json"
    if not provenance.exists():
        write_json(provenance, spec)
    elif json.loads(provenance.read_text()) != spec:
        raise ValueError("Existing provenance differs; raw history preserved")
    return target
