"""Planning helpers reject environment drift without loading native models."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def helpers() -> Any:
    path = Path("scripts/m7_dsmlp_preflight.py")
    spec = importlib.util.spec_from_file_location("m7_dsmlp_preflight", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_effective_available_respects_container_limit() -> None:
    m = helpers()
    assert m.effective_available(64 * 1024**3, 32 * 1024**3, 26 * 1024**3) == 6 * 1024**3
    assert m.effective_available(10, None, 3) == 10
    assert m.effective_available(10, 4, 8) == 0


def test_original_recipe_must_match_digest_and_actual_platform() -> None:
    m = helpers()
    recipe = {
        "runtime_platform": {"platform": "original-kernel", "machine": "x86_64"},
        "runtime_versions": {"torch": "2.14.1+cpu"},
        "encoder_code_sha256": "a" * 64,
    }
    digest = m.digest(recipe)
    env = {
        "runtime_platform": recipe["runtime_platform"],
        "runtime_versions": recipe["runtime_versions"],
        "encoder_code_sha256": "a" * 64,
    }
    m.check_recipe(recipe, digest, env)
    with pytest.raises(ValueError, match="digest"):
        m.check_recipe(recipe, "0" * 64, env)
    env = {**env, "runtime_platform": {"platform": "new-kernel", "machine": "x86_64"}}
    with pytest.raises(ValueError, match="runtime_platform"):
        m.check_recipe(recipe, digest, env)


def test_dependency_version_drift_is_never_waived() -> None:
    m = helpers()
    recipe = {
        "runtime_platform": {"platform": "kernel", "machine": "x86_64"},
        "runtime_versions": {"torch": "2.14.1+cpu"},
        "encoder_code_sha256": "a" * 64,
    }
    env = {**recipe, "runtime_versions": {"torch": "different"}}
    with pytest.raises(ValueError, match="runtime_versions"):
        m.check_recipe(recipe, m.digest(recipe), env)


def test_explicit_platform_portability_retains_dependency_checks() -> None:
    m = helpers()
    recipe = {
        "runtime_platform": {"platform": "Linux-original", "machine": "x86_64"},
        "runtime_versions": {"torch": "2.14.1+cpu"},
        "encoder_code_sha256": "a" * 64,
    }
    env = {**recipe, "runtime_platform": {"platform": "Linux-dsmlp", "machine": "x86_64"}}
    m.check_recipe(recipe, m.digest(recipe), env, allow_platform_difference=True)
    env["runtime_versions"] = {"torch": "different"}
    with pytest.raises(ValueError, match="runtime_versions"):
        m.check_recipe(recipe, m.digest(recipe), env, allow_platform_difference=True)


def test_login_node_is_rejected_even_with_kubernetes_marker() -> None:
    m = helpers()
    with pytest.raises(ValueError, match="login"):
        m.require_compute("dsmlp-login.ucsd.edu", "linux", True)
    with pytest.raises(ValueError, match="allocated"):
        m.require_compute("some-linux-node", "linux", False)
    m.require_compute("actual-compute-pod", "linux", True)


def test_execution_manifest_cannot_expand_scope(monkeypatch: Any) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    m = importlib.import_module("m7_dsmlp_parity")
    config = {
        "schema_version": 1,
        "native_math_threads": 1,
        "approved": True,
        "scope": "M7_real_model_http_parity",
        "cpus": 4,
        "ram_gb": 32,
        "gpus": 0,
        "wall_seconds": 300,
        "rss_bytes": 12 * 1024**3,
        "minimum_available_bytes": 8 * 1024**3,
    }
    m.validate_execution(config)
    for field, value in [
        ("approved", False),
        ("schema_version", 2),
        ("native_math_threads", 4),
        ("gpus", 1),
        ("wall_seconds", 301),
        ("rss_bytes", 13 * 1024**3),
        ("minimum_available_bytes", 7 * 1024**3),
    ]:
        with pytest.raises(ValueError):
            m.validate_execution({**config, field: value})


def test_supervisor_stops_its_worker_process_group(monkeypatch: Any) -> None:
    import importlib
    import subprocess
    import sys

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    m = importlib.import_module("m7_dsmlp_parity")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        m.stop_process(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_capture_preserves_actual_fixture_boundary(tmp_path: Path, monkeypatch: Any) -> None:
    import importlib

    from test_ranking_parity import configured

    from atlasrank.evaluation.ranking import offline_predict

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    m = importlib.import_module("m7_dsmlp_parity")
    context, runtime, observations, _ = configured(tmp_path)
    offline_predict(context, runtime.ranker, " ＲＥＤ apple ", [("us", "a"), ("us", "b")])
    result = m.capture(observations[0])
    assert result["dtype"] == "float32"
    assert result["keys"] == (("us", "a"), ("us", "b"))
    assert result["values"] == observations[0].values.tolist()
    assert result["missing"] == [[False] * 5] * 2
    assert result["provenance"]["schema_id"] == context.provenance.schema_id


def test_sigterm_interrupt_runs_controlled_cleanup(monkeypatch: Any) -> None:
    import os
    import signal
    import subprocess
    import sys

    import psutil

    scripts = str(Path("scripts").resolve())
    code = f"""import sys,subprocess,time
sys.path.insert(0,{scripts!r})
from m7_dsmlp_parity import termination_signals,SupervisorSignal,stop_process
with termination_signals():
 p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
 print(p.pid,flush=True)
 try:
  time.sleep(60)
 except SupervisorSignal:
  pass
 finally:
  stop_process(p)
"""
    controller = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert controller.stdout
        line = controller.stdout.readline().strip()
        assert line, "Controller must install signal cleanup before spawning work"
        child = int(line)
        os.kill(controller.pid, signal.SIGTERM)
        assert controller.wait(timeout=8) == 0
        assert (
            not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()


def test_group_cleanup_when_root_already_exited(monkeypatch: Any) -> None:
    import importlib
    import os
    import signal
    import subprocess
    import sys
    import time

    import psutil

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    m = importlib.import_module("m7_dsmlp_parity")
    root = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """import subprocess
import sys
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
print(p.pid, flush=True)
""",
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert root.stdout
    child = int(root.stdout.readline())
    root.wait(timeout=5)
    try:
        m.stop_process(root)
        for _ in range(20):
            if (
                not psutil.pid_exists(child)
                or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
            ):
                break
            time.sleep(0.05)
        assert (
            not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        try:
            os.killpg(root.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_missing_cgroup_usage_cannot_satisfy_ram_gate(monkeypatch: Any) -> None:
    m = helpers()
    monkeypatch.setattr(m, "require_compute", lambda *args: None)
    monkeypatch.setattr(m, "environment", lambda: {})
    monkeypatch.setattr(
        m,
        "resources",
        lambda: {
            "cgroup_memory_limit_bytes": 32 * 1024**3,
            "cgroup_memory_current_bytes": None,
            "effective_cpus": 4,
            "effective_available_bytes": 32 * 1024**3,
        },
    )
    report = m.inspect(None, "a" * 64)
    assert report["status"] == "blocked"
    assert "32 GB allocated RAM is not established" in report["failures"]
