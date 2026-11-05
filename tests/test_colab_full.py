"""Full-run readiness on bounded synthetic catalogs, never a GPU workload."""

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from test_dense_infrastructure import FixtureEncoder, catalog

from atlasrank.retrieval.colab_full import catalog_contract, encode_persisted, verify_coverage
from atlasrank.retrieval.colab_full_contract import approve_full


def test_approval_denies_before_any_input_work(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        approve_full(
            Path("configs/m4-colab-full-run.proposed.json"), tmp_path / "absent", tmp_path / "store"
        )
    assert not (tmp_path / "store").exists()


def test_canonical_contract_covers_final_partial_chunk(tmp_path: Path) -> None:
    source = catalog(tmp_path / "products.parquet", 10)
    contract = catalog_contract(source, 10, 4)
    assert [(c["start"], c["stop"]) for c in contract["chunks"]] == [(0, 4), (4, 8), (8, 10)]
    assert contract == catalog_contract(source, 10, 4)
    with pytest.raises(ValueError):
        catalog_contract(source, 9, 4)
    frame = pl.read_parquet(source)
    frame.reverse().write_parquet(tmp_path / "reversed.parquet")
    with pytest.raises(ValueError):
        catalog_contract(tmp_path / "reversed.parquet", 10, 4)
    frame.with_columns(pl.lit("E").alias("esci_label")).write_parquet(tmp_path / "labels.parquet")
    with pytest.raises(ValueError):
        catalog_contract(tmp_path / "labels.parquet", 10, 4)


def test_receipt_restart_no_reencoding_and_coverage(tmp_path: Path) -> None:
    source = catalog(tmp_path / "products.parquet", 10)
    store = tmp_path / "persistent"
    origin = {"scope": "synthetic"}
    encoder = FixtureEncoder(fail_call=3)
    events: list[dict[str, Any]] = []
    with pytest.raises(InterruptedError):
        encode_persisted(
            source,
            tmp_path / "first",
            encoder,
            origin,
            store,
            chunk_rows=4,
            wall_seconds=60,
            event=events.append,
        )
    assert len(list((store / "chunks").glob("*.receipt.json"))) == 1
    receipt = next((store / "chunks").glob("*.receipt.json"))
    before = receipt.read_bytes()
    resumed = FixtureEncoder()
    result = encode_persisted(
        source,
        tmp_path / "resumed",
        resumed,
        origin,
        store,
        chunk_rows=4,
        wall_seconds=60,
        event=events.append,
    )
    assert result["rows"] == 10 and resumed.calls == 3
    assert receipt.read_bytes() == before
    assert any(e["stage"] == "checkpoint_publication" for e in events)
    assert any(e["stage"] == "batch_encoding" for e in events)
    contract = catalog_contract(source, 10, 4)
    verify_coverage(tmp_path / "resumed/job", source, contract, resumed.recipe, origin)
    wrong = pl.read_parquet(source).with_columns(pl.lit("wrong").alias("product_title"))
    wrong.write_parquet(tmp_path / "wrong.parquet")
    with pytest.raises(ValueError):
        verify_coverage(
            tmp_path / "resumed/job", tmp_path / "wrong.parquet", contract, resumed.recipe, origin
        )
    clean = encode_persisted(
        source,
        tmp_path / "uninterrupted",
        FixtureEncoder(),
        origin,
        tmp_path / "clean-store",
        chunk_rows=4,
        wall_seconds=60,
    )
    assert clean == result  # includes immutable chunk IDs and job identity
    import numpy as np

    from atlasrank.retrieval.chunks import completed_chunks

    for resumed_chunk, clean_chunk in zip(
        completed_chunks(tmp_path / "resumed/job"),
        completed_chunks(tmp_path / "uninterrupted/job"),
        strict=True,
    ):
        assert resumed_chunk[1].equals(clean_chunk[1])
        assert np.array_equal(resumed_chunk[2], clean_chunk[2])
    job = json.loads((tmp_path / "resumed/job/JOB.json").read_text())
    assert job["wall_seconds"] == 60


def test_streaming_return_native_bundle_and_separate_local_budget(tmp_path: Path) -> None:
    from atlasrank.artifacts.dense import write_once
    from atlasrank.retrieval.bm25 import build as lexical_build
    from atlasrank.retrieval.colab_full_return import import_completed
    from atlasrank.retrieval.colab_full_transfer import restore_return_job, return_archive
    from atlasrank.retrieval.comparison import evaluate_development

    source = catalog(tmp_path / "products.parquet", 8)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    source.rename(prepared / "catalog.parquet")
    source = prepared / "catalog.parquet"
    dev = pl.DataFrame(
        {
            "query_id": [1],
            "query": ["apple"],
            "group_sha256": ["fixture"],
            "partition": ["development"],
            "product_locale": ["us"],
            "product_id": ["0000"],
            "esci_label": ["E"],
        }
    )
    dev.write_parquet(prepared / "development.parquet")
    from atlasrank.artifacts.dense import file_hash

    write_once(
        prepared / "audit.json",
        {
            "schema_version": 1,
            "evaluation_contract": "v0.3",
            "output_sha256": {"development.parquet": file_hash(prepared / "development.parquet")},
        },
    )
    write_once(
        prepared / "COMPLETE.json",
        {"schema_version": 1, "audit_sha256": file_hash(prepared / "audit.json")},
    )
    origin = {
        "catalog_sha256": file_hash(source),
        "prepared_audit_sha256": file_hash(prepared / "audit.json"),
    }
    encoder = FixtureEncoder()
    for k in (
        "repo",
        "revision",
        "model_files_sha256",
        "max_length",
        "padding",
        "pooling",
        "dtype",
        "attention",
    ):
        encoder.recipe[k] = "synthetic"
    encoder.recipe["device"] = "cuda:0"
    store = tmp_path / "persistent"
    encode_persisted(
        source, tmp_path / "cloud", encoder, origin, store, chunk_rows=4, wall_seconds=60
    )
    (store / "unacknowledged.zip").write_bytes(b"preserve this interrupted upload")
    exported = return_archive(store, tmp_path / "returned.zip")
    assert exported["unacknowledged_uploads"] == ["unacknowledged.zip"]
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    import shutil

    shutil.copyfile(source, inputs / "products.parquet")
    write_once(inputs / "CATALOG.json", catalog_contract(source, 8, 4))
    write_once(inputs / "REFERENCE_RECIPE.json", encoder.recipe)
    write_once(inputs / "ORIGIN.json", origin)
    ledger = tmp_path / "local-budget"
    ledger.mkdir()
    (ledger / "attempts").mkdir()
    write_once(ledger / "JOB.json", {"wall_seconds": 60})
    result = import_completed(tmp_path / "returned.zip", inputs, tmp_path / "local", ledger)
    assert result["rows"] == 8
    gpu_before = {p.name: p.read_bytes() for p in (tmp_path / "local/job/attempts").iterdir()}
    lexical_build(pl.read_parquet(source), tmp_path / "lexical", origin)
    product_recipe = dict(encoder.recipe)
    encoder.recipe = {**encoder.recipe, "device": "cpu"}
    report = evaluate_development(
        prepared,
        tmp_path / "lexical",
        tmp_path / "local/bundle",
        encoder,
        tmp_path / "report",
        {**origin, "development_sha256": file_hash(prepared / "development.parquet")},
        tmp_path / "local/job",
        product_recipe=product_recipe,
        accounting_directory=ledger,
    )
    assert report["partition"] == "development" and report["A"]["dense"]["queries"] == 1
    assert gpu_before == {
        p.name: p.read_bytes() for p in (tmp_path / "local/job/attempts").iterdir()
    }
    assert len(list((ledger / "attempts").glob("*.end.json"))) == 2
    # A changed acknowledged archive must fail before it can become an index.
    corrupt = next((store / "chunks").glob("*.zip"))
    corrupt.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        return_archive(store, tmp_path / "bad-return.zip")
    with pytest.raises(FileExistsError):
        restore_return_job(tmp_path / "returned.zip", tmp_path / "local/job")


def test_new_full_entrypoints_reject_proposals_without_side_effects(tmp_path: Path) -> None:
    from atlasrank.retrieval.colab_full_local import run as local_run
    from atlasrank.retrieval.colab_full_runner import run as cloud_run

    proposal = Path("configs/m4-colab-full-run.proposed.json")
    with pytest.raises(PermissionError):
        cloud_run(proposal, tmp_path / "inputs", tmp_path / "store", tmp_path / "scratch", 0)
    with pytest.raises(PermissionError):
        local_run(
            proposal, tmp_path / "inputs", tmp_path / "missing.zip", "wrong", tmp_path / "local"
        )
    assert not list(tmp_path.iterdir())


def test_allocation_clock_includes_setup_and_avoids_double_charge(tmp_path: Path) -> None:
    import time

    from atlasrank.retrieval.colab_runner import allocation_delay

    now = time.time()
    delay, start = allocation_delay(tmp_path, now - 30, now=now)
    assert delay == 30 and start == now - 30
    (tmp_path / "prior.end.json").write_text(json.dumps({"utc_end": now - 10}))
    delay, start = allocation_delay(tmp_path, now - 30, now=now)
    assert delay == 10 and start == now - 10
    with pytest.raises(ValueError):
        allocation_delay(tmp_path, now + 1, now=now)


def test_saved_pilot_rejoins_nonprefix_identities(tmp_path: Path) -> None:
    import numpy as np

    from atlasrank.retrieval.chunks import completed_chunks
    from atlasrank.retrieval.colab_full_return import compare_saved_pilot
    from atlasrank.retrieval.colab_transport import sha

    source = catalog(tmp_path / "products.parquet", 8)
    encode_persisted(
        source,
        tmp_path / "run",
        FixtureEncoder(),
        {},
        tmp_path / "store",
        chunk_rows=4,
        wall_seconds=60,
    )
    chunks = list(completed_chunks(tmp_path / "run/job"))
    vectors = np.concatenate([v for _, _, v in chunks])
    reference = tmp_path / "reference"
    (reference / "pilot").mkdir(parents=True)
    frame = pl.read_parquet(source)[[1, 6]]
    frame.write_parquet(reference / "pilot/products.parquet")
    np.save(reference / "pilot/cpu-vectors.npy", vectors[[1, 6]])
    (reference / "TRANSFER.json").write_text(
        json.dumps(
            {
                "files": {
                    name: sha(reference / name)
                    for name in ("pilot/products.parquet", "pilot/cpu-vectors.npy")
                }
            }
        )
    )
    report = compare_saved_pilot(tmp_path / "run/job", reference, sha(reference / "TRANSFER.json"))
    assert report["rows"] == 2 and report["component_max_absolute_error"] == 0
    frame.with_columns(pl.lit("changed").alias("product_title")).write_parquet(
        reference / "pilot/products.parquet"
    )
    with pytest.raises(ValueError):
        compare_saved_pilot(tmp_path / "run/job", reference, sha(reference / "TRANSFER.json"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("catalog_products", 4096),
        ("batch_size", 64),
        ("precision", "float16"),
        ("component_atol", 1e-3),
        ("wall_seconds", 18001),
        ("source_sha256", {}),
        ("prior_allocation_charge_approved", False),
        ("storage_approved", False),
    ],
)
def test_full_contract_frozen_gates_before_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    from atlasrank.retrieval import colab_full_contract as contract

    spec = json.loads(Path("configs/m4-colab-full-run.proposed.json").read_text())
    spec.update(
        execution_approved=True,
        runnable=True,
        storage_approved=True,
        operator_owner_token="synthetic-only",
        prior_allocation_charge_approved=True,
        persistent_destination="/content/drive/MyDrive/fixture-only",
    )
    # This is a gate-order unit fixture for the historical source snapshot.
    # Later milestones add source files; do not repin the frozen run to today's tree.
    historical_source = dict(spec["source_sha256"])
    monkeypatch.setattr(contract, "reviewed_source_hashes", lambda: historical_source)
    called = []

    def fake_verify(inputs: Path, config: dict[str, Any]) -> dict[str, Any]:
        called.append(True)
        return {}

    monkeypatch.setattr(contract, "verify_inputs", fake_verify)
    manifest = tmp_path / "synthetic.json"
    manifest.write_text(json.dumps(spec))
    contract.approve_full(manifest, tmp_path / "inputs", Path(spec["persistent_destination"]))
    assert called == [True]
    called.clear()
    spec[field] = value
    manifest.write_text(json.dumps(spec))
    with pytest.raises((ValueError, PermissionError)):
        contract.approve_full(manifest, tmp_path / "inputs", Path(spec["persistent_destination"]))
    assert not called and not (tmp_path / "inputs").exists()


def test_full_allocation_anchor_charges_gaps_across_runtimes(tmp_path: Path) -> None:
    import time

    from atlasrank.retrieval.colab_runner import allocation_delay, full_allocation_start

    now = time.time()
    assert full_allocation_start(tmp_path, now - 30) == now - 30
    # New runtime timestamp cannot discard the gap since the previous end.
    (tmp_path / "prior.end.json").write_text(json.dumps({"utc_end": now - 10}))
    origin = full_allocation_start(tmp_path, now - 3)
    assert origin == now - 30
    assert allocation_delay(tmp_path, origin, now=now)[0] == 10


def test_watchdog_stops_child_after_supervisor_kill(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    import time

    import psutil

    pidfile = tmp_path / "child.pid"
    childcode = f"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(30)"
    spec = {
        "persistent_limit_bytes": 1024**2,
        "scratch_limit_bytes": 1024**2,
        "dependency_download_limit_bytes": 1024**2,
        "rss_limit_bytes": 1024**3,
    }
    script = (
        "from pathlib import Path;import time;"
        "from atlasrank.retrieval.colab_runner import bounded_command;"
        f"bounded_command({[sys.executable, '-c', childcode]!r},"
        f"Path({str(tmp_path / 'child.log')!r}),"
        f"time.monotonic()+20,{spec!r},Path({str(tmp_path)!r}),Path({str(tmp_path)!r}),dict(__import__('os').environ))"
    )
    supervisor = subprocess.Popen([sys.executable, "-c", script], env=dict(os.environ))
    pid = None
    try:
        until = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < until:
            time.sleep(0.05)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        supervisor.kill()
        supervisor.wait(timeout=2)
        until = time.monotonic() + 2
        while (
            psutil.pid_exists(pid)
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            and time.monotonic() < until
        ):
            time.sleep(0.05)
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()
        if pid is not None and psutil.pid_exists(pid):
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass


def test_exhausted_local_budget_precedes_input_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atlasrank.retrieval import colab_full_local as local

    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "synthetic.json"
    manifest.write_text(json.dumps({"local_execution_approved": True, "runnable": True}))
    work = tmp_path / "data/esci/run"
    work.mkdir(parents=True)
    (work / "old.start.json").write_text(json.dumps({"utc_start": 0, "reserved_seconds": 7200}))
    (work / "old.end.json").write_text(json.dumps({"elapsed_seconds": 7200}))

    def unexpected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Must not hash inputs after budget is exhausted")

    monkeypatch.setattr(local, "approve_full", unexpected)
    with pytest.raises(RuntimeError, match="exhausted"):
        local.run(manifest, tmp_path / "missing", tmp_path / "missing.zip", "unused", work)


def test_failed_local_preflight_is_charged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from atlasrank.retrieval import colab_full_local as local
    from atlasrank.retrieval.colab_runner import charged_seconds

    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "synthetic.json"
    manifest.write_text(json.dumps({"local_execution_approved": True, "runnable": True}))
    work = tmp_path / "data/esci/run"
    monkeypatch.setattr(local, "approve_full", lambda *a, **kw: {})
    with pytest.raises(FileNotFoundError):
        local.run(manifest, tmp_path / "inputs", tmp_path / "missing.zip", "unused", work)
    ends = list(work.glob("*.end.json"))
    assert len(ends) == 1 and json.loads(ends[0].read_text())["status"] == "failed"
    assert charged_seconds(work) > 0


def test_cloud_payload_preflight_is_deferred_to_accounted_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atlasrank.retrieval import colab_full_runner as cloud

    calls = []

    def metadata_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["verify_payload"] is False
        calls.append("metadata-only")
        return {}

    def supervised(*args: Any, **kwargs: Any) -> None:
        calls.append("accounted-session")

    monkeypatch.setattr(cloud, "approve_full", metadata_gate)
    monkeypatch.setattr(cloud, "execute_session", supervised)
    cloud.run(tmp_path / "manifest", tmp_path / "inputs", tmp_path / "store", tmp_path / "work", 1)
    assert calls == ["metadata-only", "accounted-session"]
    assert not list(tmp_path.iterdir())
