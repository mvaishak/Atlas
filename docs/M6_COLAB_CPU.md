# M6 Colab CPU: exact native-boundary execution cells

Verified run: **42 passed, zero skipped**, using CPU runtime **2026.07**, Python 3.12.13. Latest was observed as Python 3.13 and refused. Choose CPU and a compatible Python 3.12 runtime before running.

Upload `notebooks/atlasrank_m6_xgboost_cpu.ipynb` to your private Colab account, then run its five code cells in order. The source ZIP is `.local/colab/atlasrank-m6-xgboost-cpu-v2.zip` (131,423 bytes). Its SHA-256 is `003c50fb8a1b5d361443ad7f063e1207207fe5a4642fd9e593f4304ef3fbd21b`.

No Drive mount is required. If any cell fails, run cell 5 to return its available evidence. Do not change pins or skip a guard. Use a fresh CPU runtime/directory for a new execution; preserve previous outputs.

## Cell 1

```python
import hashlib
import json
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

assert sys.platform == "linux" and sys.version_info[:2] == (3, 12), sys.version
gpu = shutil.which("nvidia-smi")
if gpu:
    probe = subprocess.run([gpu, "-L"], capture_output=True, text=True, timeout=10)
    assert not (probe.returncode == 0 and probe.stdout.strip()), "Select a CPU runtime"
BASE = Path("/content/atlasrank-m6-cpu-v1")
BASE.mkdir(exist_ok=False)
EVIDENCE = BASE / "evidence"
EVIDENCE.mkdir()
discovery = {
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "accelerator": "none",
}
(EVIDENCE / "discovery.json").write_text(json.dumps(discovery, indent=2))
print(discovery)```

## Cell 2

```python
from google.colab import files

uploaded = files.upload()
assert len(uploaded) == 1, "Upload only the AtlasRank source ZIP"
blob = next(iter(uploaded.values()))
assert len(blob) < 4 * 1024**2
EXPECTED_SHA256 = "003c50fb8a1b5d361443ad7f063e1207207fe5a4642fd9e593f4304ef3fbd21b"
assert hashlib.sha256(blob).hexdigest() == EXPECTED_SHA256, "Wrong archive hash"
archive_path = BASE / "source.zip"
archive_path.write_bytes(blob)
SOURCE = BASE / "source"
SOURCE.mkdir()
with zipfile.ZipFile(archive_path) as archive:
    manifest = json.loads(archive.read("manifest.json"))
    assert len(archive.namelist()) == len(set(archive.namelist()))
    assert set(archive.namelist()) == set(manifest["files"]) | {"manifest.json"}
    assert sum(i.file_size for i in archive.infolist()) < 8 * 1024**2
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        assert not path.is_absolute() and ".." not in path.parts
        assert not stat.S_ISLNK(item.external_attr >> 16)
        content = archive.read(item)
        if item.filename != "manifest.json":
            pin = manifest["files"][item.filename]
            assert len(content) == pin["bytes"]
            assert hashlib.sha256(content).hexdigest() == pin["sha256"]
        target = SOURCE.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
(EVIDENCE / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
(EVIDENCE / "input-archive.sha256").write_text(EXPECTED_SHA256 + "\n")
print("Verified", len(manifest["files"]), "source/config files; no data/model artifacts")```

## Cell 3

```python
VENV = BASE / "venv"
subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(VENV)], check=True, timeout=120)
PY = str(VENV / "bin/python")
pip = [sys.executable, "-m", "pip", "--python", PY]
command = pip + [
    "install",
    "--disable-pip-version-check",
    "--no-cache-dir",
    "--only-binary=:all:",
    "--index-url",
    "https://pypi.org/simple",
    "-r",
    str(SOURCE / "requirements-test.txt"),
    "-c",
    str(SOURCE / "constraints-test.txt"),
    "--report",
    str(EVIDENCE / "pip-install-report.json"),
]
with (EVIDENCE / "install.log").open("w") as log:
    installed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
print((EVIDENCE / "install.log").read_text()[-8000:])
installed.check_returncode()  # Never loosen a pin after installation failure.
checked = subprocess.run(pip + ["check"], capture_output=True, text=True, timeout=30)
(EVIDENCE / "pip-check.log").write_text(checked.stdout + checked.stderr)
checked.check_returncode()
frozen = subprocess.run(
    pip + ["freeze", "--all"], capture_output=True, text=True, check=True, timeout=30
)
(EVIDENCE / "installed.txt").write_text(frozen.stdout)
print("Isolated test environment ready; Colab native Torch was not installed or changed")```

## Cell 4

```python
command = [
    PY,
    str(SOURCE / "scripts/check_xgboost_boundary.py"),
    "--output",
    str(EVIDENCE / "native-results"),
]
with (EVIDENCE / "runner.log").open("w") as log:
    run = subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, timeout=210)
print((EVIDENCE / "runner.log").read_text())
result_path = EVIDENCE / "native-results/validation.json"
result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "missing"}
print(json.dumps(result, indent=2))
print("Native gate passed:", run.returncode == 0 and result.get("status") == "passed")
# Always run the download cell, including after a failed native check.```

## Cell 5

```python
checksum_path = EVIDENCE / "checksums.json"
checksums = {
    str(p.relative_to(EVIDENCE)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(EVIDENCE.rglob("*"))
    if p.is_file() and p != checksum_path
}
checksum_path.write_text(json.dumps(checksums, indent=2) + "\n")
returned = Path("/content/atlasrank-m6-boundary-results-v1.zip")
assert not returned.exists(), "Preserve previous return archives"
shutil.make_archive(str(returned.with_suffix("")), "zip", EVIDENCE)
print("Return ZIP SHA-256:", hashlib.sha256(returned.read_bytes()).hexdigest())
print("Return ZIP bytes:", returned.stat().st_size)
files.download(str(returned))```

The runner requires 42 or more cases with no skips/failures/errors and `native-boundary.json`. A skipped Mac fixture suite does not pass this native gate. Verify the downloaded ZIP checksums and source pins. See [the execution review](reviews/M6.colab-boundary.md). Actual M4/M5 bundle parity and fitting approval remain separate.

To rebuild from the same reviewed working-tree source snapshot, preserve existing outputs and use new paths:

```sh
.venv/bin/python scripts/package_xgboost_colab.py --output .local/colab/m6-next-source.zip --notebook .local/colab/m6-next-notebook.ipynb
```

The builder records current HEAD and exact working-tree file bytes. A different source/commit produces a new archive/hash; never relabel it as the historical verified v2 archive.
