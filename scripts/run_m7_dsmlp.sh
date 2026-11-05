#!/usr/bin/env bash
# Run only inside the user's already allocated DSMLP CPU container.
set -euo pipefail
umask 077
PORTABILITY_ARGS=()
case "${1:-}" in
    "") ;;
    --allow-platform-difference) PORTABILITY_ARGS=(--allow-platform-difference); shift ;;
    *) echo "Usage: $0 [--allow-platform-difference]" >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || { echo "Unexpected runner arguments" >&2; exit 2; }
UPDATE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$UPDATE/.." && pwd)
BASE_PYTHON=/opt/conda/bin/python
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
STAMP=$(date -u +%Y%m%dT%H%M%SZ)-$$
SETUP="$ROOT/evidence/setup-v2-$STAMP"
OUTPUT="$ROOT/evidence/real-http-v2-$STAMP"
mkdir -p "$SETUP"
exec > >(tee "$SETUP/setup.log") 2>&1
trap 'code=$?; printf "exit_code=%s\nfinished_utc=%s\n" "$code" "$(date -u +%FT%TZ)" > "$SETUP/exit.txt"' EXIT
printf 'started_utc=%s\nupdate=%s\noutput=%s\n' "$(date -u +%FT%TZ)" "$UPDATE" "$OUTPUT"

# Standard-library checks precede downloads and package installation.
"$BASE_PYTHON" - "$ROOT" "$SETUP" "$UPDATE" "${PORTABILITY_ARGS[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import socket
import sys

root, setup, update = map(Path, sys.argv[1:4])
sys.path.insert(0, str(update / "scripts"))
execution = json.loads((update / "approved-execution.json").read_text())
runpy.run_path(str(update / "scripts/m7_dsmlp_parity.py"))["validate_execution"](execution)
if sys.argv[4:] and execution.get("compatibility_policy") != "record_linux_platform_difference":
    raise SystemExit("Execution manifest must declare platform portability")
hostname = socket.gethostname()
if sys.platform != "linux" or "login" in hostname.lower():
    raise SystemExit("Run inside the allocated Linux CPU container only")
if not os.environ.get("KUBERNETES_SERVICE_HOST"):
    raise SystemExit("Allocated Kubernetes container is not established")
if platform.machine() != "x86_64":
    raise SystemExit("Expected x86_64 compute container")

def integer(paths):
    for name in paths:
        path = Path(name)
        if path.exists():
            value = path.read_text().strip()
            return None if value == "max" else int(value)
    return None

limit = integer(("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"))
used = integer(("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"))
memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
available = int(memory["MemAvailable"].split()[0]) * 1024
if limit is None or used is None or not 32 * 10**9 <= limit <= 32 * 1024**3:
    raise SystemExit("Expected a bounded 32 GB RAM allocation")
effective = min(available, max(0, limit - used))
cpu_max = Path("/sys/fs/cgroup/cpu.max")
if cpu_max.exists():
    quota, period = cpu_max.read_text().split()
    cpus = None if quota == "max" else int(quota) / int(period)
else:
    quota = integer(("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",))
    period = integer(("/sys/fs/cgroup/cpu/cpu.cfs_period_us",))
    cpus = quota / period if quota and quota > 0 and period else None
affinity = len(os.sched_getaffinity(0))
effective_cpus = min(affinity, cpus) if cpus is not None else affinity
report = dict(hostname=hostname, platform=platform.platform(), python=sys.version,
              memory_limit_bytes=limit, effective_available_bytes=effective,
              effective_cpus=effective_cpus)
(setup / "bootstrap-precheck.json").write_text(json.dumps(report, indent=2) + "\n")
if effective_cpus != 4 or effective < 8 * 1024**3:
    raise SystemExit("Need exactly 4 allocated CPUs and at least 8 GiB effective available RAM")
inventory = json.loads((root / "metadata/transfer-inventory.json").read_text())
records = inventory["component_files"]
for record in records:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit("Unsafe inventory path")
    path = root / "artifacts" / relative
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
        raise SystemExit("Immutable artifact size/type mismatch: " + str(relative))
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise SystemExit("Immutable artifact checksum mismatch: " + str(relative))
(setup / "artifact-verification.json").write_text(json.dumps(
    dict(status="passed", files=len(records), inventory_sha256=hashlib.sha256(
        (root / "metadata/transfer-inventory.json").read_bytes()).hexdigest()), indent=2) + "\n")
print("Compute resources and immutable input checks passed", flush=True)
manifest = json.loads((update / "update-manifest.json").read_text())
for record in manifest["files"]:
    relative = Path(record["path"])
    path = update / relative
    if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
        raise SystemExit("Unsafe update path")
    if path.stat().st_size != record["bytes"]:
        raise SystemExit("Update size mismatch: " + str(relative))
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit("Update hash mismatch: " + str(relative))
(setup / "update-verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Updated code and supplied metadata hashes passed", flush=True)
PY

BOOTSTRAP="$ROOT/bootstrap-v2"
ENVIRONMENT="$ROOT/env-v2"
if [[ ! -e "$BOOTSTRAP" ]]; then
    "$BASE_PYTHON" -m venv "$BOOTSTRAP"
fi
"$BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check --index-url https://pypi.org/simple 'uv==0.8.22'
UV="$BOOTSTRAP/bin/uv"
[[ "$("$UV" --version)" == 'uv 0.8.22' ]] || { echo 'Unexpected uv version'; exit 1; }
export UV_PYTHON_INSTALL_DIR="$ROOT/python-v2" UV_CACHE_DIR="$ROOT/uv-cache-v2"
"$UV" python install 3.12
if [[ ! -e "$ENVIRONMENT" ]]; then
    "$UV" venv --python 3.12 "$ENVIRONMENT"
fi
PYTHON="$ENVIRONMENT/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"$UV" pip install --python "$PYTHON" --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
# Every remaining distribution is pinned to recovered M6 versions, plus approved uvicorn.
# No dependency solver may introduce additional versions silently.
"$UV" pip install --python "$PYTHON" --no-deps --index-url https://pypi.org/simple \
    -r "$UPDATE/requirements-dsmlp-cpu-v1.txt"
"$UV" pip check --python "$PYTHON"
"$PYTHON" - "$UPDATE/src" <<'PY'
from pathlib import Path
import site
import sys
source = Path(sys.argv[1]).resolve()
if not (source / "atlasrank/__init__.py").is_file():
    raise SystemExit("Updated atlasrank source is missing")
(Path(site.getsitepackages()[0]) / "atlasrank-m7-update-v2.pth").write_text(str(source) + "\n")
PY
"$UV" pip freeze --python "$PYTHON" > "$SETUP/packages.txt"
"$PYTHON" -c 'import platform, sys; print(sys.version); print(sys.executable); print(platform.platform())' > "$SETUP/python.txt"
sha256sum "$UPDATE/requirements-dsmlp-cpu-v1.txt" "$UPDATE/approved-execution.json" > "$SETUP/setup-inputs.sha256"

# Model execution is offline and remains bounded by the harness's existing limits.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
cd "$UPDATE"
"$PYTHON" scripts/m7_dsmlp_parity.py \
    --execution-manifest "$UPDATE/approved-execution.json" \
    --descriptor "$ROOT/artifacts/.local/m7-v1/serving-descriptor-v2" \
    --artifacts "$ROOT/artifacts" \
    --original-query-recipe "$UPDATE/reconstructed-m6-query-recipe-v1.json" \
    "${PORTABILITY_ARGS[@]}" \
    --output "$OUTPUT"
printf 'Evidence saved in %s\nSetup saved in %s\n' "$OUTPUT" "$SETUP"
