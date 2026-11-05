"""Local synthetic checks only. No CUDA, Drive, downloads or model inference."""

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from test_dense_infrastructure import FixtureEncoder, build_fixture, catalog

from atlasrank.artifacts.dense import digest, file_hash
from atlasrank.evaluation.cuda_checks import compare_cuda_vectors
from atlasrank.retrieval.chunks import encode_chunks
from atlasrank.retrieval.colab_environment import approval, validate_installed, validate_runtime
from atlasrank.retrieval.colab_pilot import persist_metadata, restore_job
from atlasrank.retrieval.colab_runner import charged_seconds, run
from atlasrank.retrieval.colab_transport import extract_inputs, publish, restore, safe_extract
from atlasrank.retrieval.cuda_encoder import CudaEncoder, gpu_report, token_arrays, token_hashes


def test_unapproved_pilot_and_full_fail_before_work(tmp_path: Path) -> None:
    for path in ("configs/m4-colab-pilot.proposed.json", "configs/m4-colab-full-run.proposed.json"):
        with pytest.raises(PermissionError, match="explicit"):
            run(Path(path), tmp_path / "missing", tmp_path / "store", tmp_path / "work")
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("change", ["precision", "titles", "wall_seconds", "vram_limit_bytes"])
def test_unreviewed_configuration_rejected(tmp_path: Path, change: str) -> None:
    spec = json.loads(Path("configs/m4-colab-pilot.proposed.json").read_text())
    spec.update(execution_approved=True)
    spec[change] = "unapproved"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="configuration"):
        approval(path, tmp_path, tmp_path / "store")


def test_missing_storage_approval_rejected(tmp_path: Path) -> None:
    spec = json.loads(Path("configs/m4-colab-pilot.proposed.json").read_text())
    spec["execution_approved"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(PermissionError, match="Drive"):
        approval(path, tmp_path, tmp_path / "store")


def test_transport_interruption_receipts_and_immutable_repeat(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "vectors.npy").write_bytes(b"synthetic bytes")
    store = tmp_path / "store"
    store.mkdir()
    (store / "chunk.partial.zip").write_bytes(b"abandoned transfer")
    receipt = publish(source, store, "chunk", 1024**2)
    before = {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in store.iterdir()}
    assert publish(source, store, "chunk", 1024**2) == receipt
    assert before == {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in store.iterdir()}
    restore(store, "chunk", tmp_path / "restored", 1024**2)
    assert (tmp_path / "restored/vectors.npy").read_bytes() == b"synthetic bytes"
    (store / receipt["archive"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore(store, "chunk", tmp_path / "rejected", 1024**2)


@pytest.mark.parametrize("kind", ["traversal", "duplicate", "oversize", "symlink"])
def test_unsafe_archive_rejected(tmp_path: Path, kind: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        if kind == "traversal":
            z.writestr("../outside", "x")
        elif kind == "duplicate":
            z.writestr("same", "x")
            with pytest.warns(UserWarning):
                z.writestr("same", "y")
        elif kind == "oversize":
            z.writestr("file", "x" * 101)
        else:
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            z.writestr(info, "outside")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out", 100)
    assert not (tmp_path / "out").exists()


def test_input_hash_gate_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "input.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="reviewed"):
        extract_inputs(archive, tmp_path / "out", "wrong")
    assert not (tmp_path / "out").exists()


def test_token_ids_hashes_batch_independent_and_normalization() -> None:
    class Tokenizer:
        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {
                "padding": "max_length",
                "truncation": True,
                "max_length": 256,
                "return_tensors": "np",
            }
            return {
                "input_ids": np.array([[len(t)] * 256 for t in texts]),
                "attention_mask": np.ones((len(texts), 256), dtype=np.int64),
            }

    tokenizer = Tokenizer()
    texts = [" ＡPPLE\t juice ", "", "Straße"]
    hashes = token_hashes(token_arrays(tokenizer, texts))
    assert hashes == [token_hashes(token_arrays(tokenizer, [t]))[0] for t in texts]
    assert hashes[0] == token_hashes(token_arrays(tokenizer, ["apple juice"]))[0]
    altered = token_arrays(tokenizer, texts)
    altered["attention_mask"][0, 0] = 0
    assert token_hashes(altered)[0] != hashes[0]


def test_gpu_unavailable_and_workspace_fail_before_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Torch:
        cuda = NoCuda()

    with pytest.raises(RuntimeError, match="CUDA GPU"):
        gpu_report(Torch())
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="Fresh worker"):
        CudaEncoder({}, {})


def test_environment_native_cpu_or_wrong_torch_rejected() -> None:
    environment = json.loads(Path("environments/colab/environment.json").read_text())
    for info in (
        {"torch_version": "2.0.0", "torch_cuda": None, "cuda_available": False},
        {"torch_version": "2.14.1", "torch_cuda": None, "cuda_available": False},
    ):
        with pytest.raises(RuntimeError, match="Unreviewed"):
            validate_runtime(info, environment)
    assert validate_installed(environment) == environment["pinned_packages"]
    requirements = Path("environments/colab/requirements.txt").read_text()
    assert "\ntorch==" not in requirements
    assert "--hash=sha256:" in requirements


def test_budget_recovery_includes_downtime_and_never_resets(tmp_path: Path) -> None:
    (tmp_path / "old.start.json").write_text(json.dumps({"utc_start": 0, "reserved_seconds": 600}))
    assert charged_seconds(tmp_path) == 600
    assert charged_seconds(tmp_path) == 600
    assert (tmp_path / "old.recovered.json").exists()


def test_cross_device_tolerances_report_near_ties_and_reject_errors() -> None:
    rng = np.random.default_rng(42)
    cpu = rng.normal(size=(128, 8)).astype(np.float32)
    cpu /= np.linalg.norm(cpu, axis=1, keepdims=True)
    cpu[1] = cpu[2]
    cuda = cpu.copy()
    cuda[1, 0] += 1e-7
    cuda /= np.linalg.norm(cuda, axis=1, keepdims=True)
    report = compare_cuda_vectors(cpu, cuda)
    assert report["component_max_absolute_error"] < 1e-5
    assert report["quality_evidence"] is False
    assert report["m6_exact_feature_parity_waived"] is False
    bad = cpu.copy()
    bad[0] = -bad[0]
    with pytest.raises(AssertionError):
        compare_cuda_vectors(cpu, bad)
    with pytest.raises(ValueError):
        compare_cuda_vectors(cpu, cuda[:-1])


def test_chunk_callback_failure_resumes_without_reencoding(tmp_path: Path) -> None:
    source = catalog(tmp_path / "catalog.parquet")
    job = tmp_path / "job"
    store = tmp_path / "store"
    staging = tmp_path / "staging"
    staging.mkdir()
    first = FixtureEncoder()

    def interrupt(chunk: Path) -> None:
        persist_metadata(job, store, staging, 1024**2)
        publish(chunk, store / "chunks", chunk.name, 1024**2)
        raise InterruptedError("After persisted first chunk")

    with pytest.raises(InterruptedError):
        encode_chunks(
            source,
            job,
            first,
            {"scope": "fixture"},
            chunk_rows=4,
            wall_seconds=60,
            on_chunk=interrupt,
        )
    persist_metadata(job, store, staging, 1024**2)
    restore_job(store, tmp_path / "fresh-runtime-job", staging, 1024**2)
    fresh = FixtureEncoder()
    callbacks = []
    encode_chunks(
        source,
        tmp_path / "fresh-runtime-job",
        fresh,
        {"scope": "fixture"},
        chunk_rows=4,
        wall_seconds=60,
        on_chunk=lambda p: callbacks.append(p.name),
    )
    assert fresh.calls == 2  # Four missing rows only; first four restored rows skipped.
    assert callbacks == ["000000000000", "000000000004"]
    changed = FixtureEncoder()
    changed.recipe["device"] = "cuda:0"
    with pytest.raises(ValueError, match="Resume"):
        encode_chunks(
            source,
            tmp_path / "fresh-runtime-job",
            changed,
            {"scope": "fixture"},
            chunk_rows=4,
            wall_seconds=60,
        )


def test_explicit_product_query_recipe_preserved(tmp_path: Path) -> None:
    from atlasrank.retrieval.comparison import evaluate_development
    from atlasrank.retrieval.dense_bundle import build_bundle

    job, _, encoder = build_fixture(tmp_path)
    # Synthetic keys satisfy semantic comparison; device difference remains explicit.
    product_recipe = {**encoder.recipe, "device": "cuda:0"}
    product_job = json.loads((job / "JOB.json").read_text())
    # Rebuild a separate fixture with CUDA-tagged synthetic encoder, no actual GPU work.
    encoder.recipe = product_recipe
    tagged = tmp_path / "tagged-job"
    encode_chunks(
        tmp_path / "catalog.parquet",
        tagged,
        encoder,
        {"scope": "synthetic_fixture"},
        chunk_rows=4,
        wall_seconds=60,
    )
    build_bundle(tagged, tmp_path / "tagged-bundle")
    encoder.recipe = {**product_recipe, "device": "cpu"}
    # Confirm backend pairing reaches source validation; incompatible semantics fail first
    # after the source files are pinned. Full successful wrapper coverage is in M4 fixtures.
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for n in ("audit.json", "catalog.parquet", "development.parquet"):
        (prepared / n).write_bytes(b"fixture")
    source = {
        "prepared_audit_sha256": file_hash(prepared / "audit.json"),
        "catalog_sha256": file_hash(prepared / "catalog.parquet"),
        "development_sha256": file_hash(prepared / "development.parquet"),
    }
    with pytest.raises(ValueError, match="semantics"):
        evaluate_development(
            prepared,
            tmp_path / "lexical",
            tmp_path / "tagged-bundle",
            encoder,
            tmp_path / "report",
            source,
            tagged,
            product_recipe={**product_recipe, "pooling": "unapproved"},
        )
    assert product_job["recipe"]["device"] == "cpu"
    assert digest(product_recipe) != digest(encoder.recipe)


def test_notebook_thin_unexecuted_and_compiles() -> None:
    nb = json.loads(Path("notebooks/atlasrank_m4_colab.ipynb").read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for cell in cells:
        assert cell["execution_count"] is None and cell["outputs"] == []
        source = "".join(cell["source"])
        compile(source, "notebook", "exec")
        assert "model(" not in source and "encode(" not in source
        assert "pip install torch" not in source
    assert "colab_runner" in "".join("".join(c["source"]) for c in cells)


def test_cuda_adapter_batch_timing_and_tokens_with_tensor_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    class Tensor:
        def __init__(self, values: Any) -> None:
            self.values = values

        def to(self, device: str) -> "Tensor":
            assert device == "cuda:0"
            return self

        def detach(self) -> "Tensor":
            return self

        def cpu(self) -> "Tensor":
            return self

        def numpy(self) -> Any:
            return self.values

    class Tokenizer:
        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            return {
                "input_ids": np.ones((len(texts), 256), dtype=np.int64),
                "attention_mask": np.ones((len(texts), 256), dtype=np.int64),
            }

    class Model:
        tokenizer = Tokenizer()

        def __call__(self, features: dict[str, Any]) -> dict[str, Any]:
            vector = np.ones((len(features["input_ids"].values), 384), dtype=np.float32)
            vector /= np.linalg.norm(vector, axis=1, keepdims=True)
            return {"sentence_embedding": Tensor(vector)}

    syncs: list[int] = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: syncs.append(device)),
        from_numpy=Tensor,
        inference_mode=nullcontext,
        autocast=lambda **kwargs: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    encoder = CudaEncoder.__new__(CudaEncoder)
    encoder.model = Model()
    encoder.timings = []
    monkeypatch.setattr(encoder, "check_memory", lambda: {})
    values = encoder.encode(["apple", ""])
    assert values.shape == (2, 384) and len(syncs) == 4
    assert encoder.last_token_hashes == token_hashes(token_arrays(Model.tokenizer, ["apple", ""]))
    assert encoder.timings[0]["model_inference_seconds"] >= 0
    assert encoder.timings[0]["titles"] == 2


def test_cuda_products_cpu_queries_successful_fixture(tmp_path: Path) -> None:
    from atlasrank.artifacts.dense import write_once
    from atlasrank.retrieval.bm25 import build as build_lexical
    from atlasrank.retrieval.comparison import evaluate_development
    from atlasrank.retrieval.dense_bundle import build_bundle

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    source = catalog(prepared / "catalog.parquet", 8)
    development = pl.DataFrame(
        {
            "query_id": [1],
            "query": ["apple"],
            "product_locale": ["us"],
            "product_id": ["0000"],
            "esci_label": ["E"],
            "group_sha256": ["fixture"],
            "partition": ["development"],
        }
    )
    development.write_parquet(prepared / "development.parquet")
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
    pins = {
        "catalog_sha256": file_hash(source),
        "prepared_audit_sha256": file_hash(prepared / "audit.json"),
        "development_sha256": file_hash(prepared / "development.parquet"),
    }
    encoder = FixtureEncoder()
    for field in (
        "repo",
        "revision",
        "model_files_sha256",
        "max_length",
        "padding",
        "pooling",
        "dtype",
        "attention",
    ):
        encoder.recipe[field] = "synthetic-fixture-only"
    encoder.recipe["device"] = "cuda:0"
    product_recipe = dict(encoder.recipe)
    job = tmp_path / "job"
    encode_chunks(source, job, encoder, pins, chunk_rows=4, wall_seconds=60)
    build_bundle(job, tmp_path / "bundle")
    build_lexical(pl.read_parquet(source), tmp_path / "lexical", pins)
    encoder.recipe = {**encoder.recipe, "device": "cpu"}
    report = evaluate_development(
        prepared,
        tmp_path / "lexical",
        tmp_path / "bundle",
        encoder,
        tmp_path / "report",
        pins,
        job,
        product_recipe=product_recipe,
    )
    assert report["provenance"]["query_recipe"]["device"] == "cpu"
    assert report["provenance"]["product_recipe"]["device"] == "cuda:0"
    assert report["provenance"]["recipe_id"] != report["provenance"]["product_recipe_id"]
    assert report["partition"] == "development" and report["test_judgments_accessed"] is False


def test_approved_manifest_source_and_transfer_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import atlasrank.retrieval.colab_environment as module

    spec = json.loads(Path("configs/m4-colab-pilot.proposed.json").read_text())
    destination = Path("/content/drive/MyDrive/AtlasRank/test-approved")
    spec.update(
        execution_approved=True,
        storage_approved=True,
        persistent_destination=str(destination),
        operator_owner_token="fixture",
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "title.txt").write_text("synthetic title")
    transfer = {
        "kind": "m4_cpu_exact_label_free_pilot",
        "rows": 4096,
        "files": {"title.txt": file_hash(inputs / "title.txt")},
    }
    (inputs / "TRANSFER.json").write_text(json.dumps(transfer))
    spec["transfer_manifest_sha256"] = file_hash(inputs / "TRANSFER.json")
    spec["source_sha256"] = {"fixture": "pin"}
    monkeypatch.setattr(module, "reviewed_source_hashes", lambda: {"fixture": "pin"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec))
    assert approval(manifest, inputs, destination) == spec
    (inputs / "title.txt").write_text("changed")
    with pytest.raises(ValueError, match="transferred"):
        approval(manifest, inputs, destination)
    spec["source_sha256"] = {}
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Reviewed source"):
        approval(manifest, inputs, destination)


def test_cuda_hardware_and_memory_counters_with_test_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    import sys
    from types import SimpleNamespace

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda i: SimpleNamespace(name="NVIDIA L4"),
        mem_get_info=lambda i: (8 * 1024**3, 23 * 1024**3),
        get_device_capability=lambda i: (8, 9),
        memory_allocated=lambda i: 1024,
        memory_reserved=lambda i: 2048,
        max_memory_allocated=lambda i: 4096,
        max_memory_reserved=lambda i: 8192,
    )
    torch = SimpleNamespace(
        cuda=cuda,
        __version__="2.14.1+cu130",
        version=SimpleNamespace(cuda="13.0"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 92400)),
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="580.82.07\n"))
    info = gpu_report(torch)
    assert info["driver_version"] == "580.82.07" and info["compute_capability"] == [8, 9]
    monkeypatch.setitem(sys.modules, "torch", torch)
    encoder = CudaEncoder.__new__(CudaEncoder)
    encoder.vram_limit = 16384
    assert encoder.check_memory()["peak_reserved_bytes"] == 8192
    encoder.vram_limit = 1024
    with pytest.raises(RuntimeError, match="memory"):
        encoder.check_memory()


def test_runtime_success_and_native_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import platform
    import sys

    monkeypatch.setattr(sys, "version_info", (3, 13, 16))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    environment = json.loads(Path("environments/colab/environment.json").read_text())
    info = {
        "torch_version": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "cuda_available": True,
        "gpu_name": "NVIDIA L4",
        "device_count": 1,
    }
    validate_runtime(info, environment)
    with pytest.raises(RuntimeError, match="Unreviewed"):
        validate_runtime({**info, "torch_version": "2.9.0+cu128"}, environment)


def test_subprocess_supervisor_records_success_and_stops_at_deadline(tmp_path: Path) -> None:
    import os
    import sys
    import time

    from atlasrank.retrieval.colab_runner import bounded_command

    spec = json.loads(Path("configs/m4-colab-pilot.proposed.json").read_text())
    store, work = tmp_path / "store", tmp_path / "work"
    store.mkdir()
    work.mkdir()
    result = bounded_command(
        [sys.executable, "-c", "print('synthetic')"],
        work / "ok.log",
        time.monotonic() + 5,
        spec,
        store,
        work,
        dict(os.environ),
    )
    assert result["elapsed_seconds"] > 0 and (work / "ok.log").read_text().strip() == "synthetic"
    with pytest.raises(TimeoutError, match="cumulative"):
        bounded_command(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            work / "stopped.log",
            time.monotonic() + 0.1,
            spec,
            store,
            work,
            dict(os.environ),
        )
