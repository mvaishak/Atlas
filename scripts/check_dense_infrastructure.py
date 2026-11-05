"""Bounded resume/bundle/evaluator integration using 64 already-encoded pilot rows."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from atlasrank.artifacts.dense import digest, file_hash, write_once
from atlasrank.data.common import check_disk
from atlasrank.retrieval.bm25 import build as build_lexical
from atlasrank.retrieval.chunks import accounted_seconds, completed_chunks, encode_chunks
from atlasrank.retrieval.comparison import evaluate_development
from atlasrank.retrieval.dense_bundle import BUNDLE_FILES, build_bundle, load_bundle


class ReplayEncoder:
    def __init__(
        self,
        titles: list[str],
        vectors: NDArray[np.float32],
        source_hash: str,
        fail: bool = False,
        pause_marker: Path | None = None,
    ) -> None:
        self.recipe: dict[str, Any] = {
            "kind": "saved_pilot_vector_replay_fixture",
            "dimensions": 384,
            "batch_size": 32,
            "device": "replay_no_inference",
            "source_vectors_sha256": source_hash,
        }
        self.by_title = dict(zip(titles, vectors, strict=True))
        self.fail = fail
        self.calls = 0
        self.pause_marker = pause_marker

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        self.calls += 1
        if self.fail and self.calls == 2:
            raise InterruptedError("Injected interruption before the second chunk completes")
        if self.pause_marker is not None and self.calls == 2:
            write_once(self.pause_marker, {"ready_for_parent_termination": True})
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                time.sleep(0.1)
            raise RuntimeError("Parent didn't terminate the fixture child within 10 seconds")
        return np.asarray([self.by_title[text] for text in texts], dtype=np.float32)


def crash_child(output: Path) -> None:
    control = json.loads((output / "child-control.json").read_text())
    source = output / "fixture-prepared/catalog.parquet"
    products = pl.read_parquet(source)
    if products.height != 64:
        raise ValueError("Crash fixture is limited to 64 replay rows")
    encoder = ReplayEncoder(
        products["product_title"].to_list(),
        np.load(output / "replay.npy", allow_pickle=False),
        control["source_hash"],
        pause_marker=output / "child-ready.json",
    )
    encode_chunks(
        source,
        output / "resumed",
        encoder,
        control["origin"],
        chunk_rows=32,
        wall_seconds=120,
        budget_root=Path("data/esci"),
    )


def run(pilot: Path, output: Path, process_interrupt: bool = False) -> None:
    root = Path("data/esci")
    if output.exists():
        raise FileExistsError("Preserve earlier integration history")
    if not all(p.resolve().is_relative_to(root.resolve()) for p in (pilot, output)):
        raise ValueError("Integration artifacts must remain in the budgeted data root")
    report = json.loads((pilot / "report.json").read_text())
    if (
        file_hash(pilot / "report.json")
        != json.loads((pilot / "COMPLETE.json").read_text())["report_sha256"]
    ):
        raise ValueError("Pilot completion differs")
    for name, field in (("sample.parquet", "sample_sha256"), ("vectors.npy", "vectors_sha256")):
        if file_hash(pilot / name) != report[field]:
            raise ValueError("Pilot component differs")
    check_disk(root, 32 * 1024**2)
    output.mkdir(parents=True)
    prepared = output / "fixture-prepared"
    prepared.mkdir()
    frame = pl.read_parquet(pilot / "sample.parquet").with_row_index("source_row")
    selected = frame.unique(subset="product_title", keep="first", maintain_order=True).head(64)
    products = selected.select("product_locale", "product_id", "product_title")
    vectors = np.asarray(
        np.load(pilot / "vectors.npy", allow_pickle=False, mmap_mode="r")[
            selected["source_row"].to_list()
        ],
        dtype=np.float32,
    )
    assert len(vectors) == 64
    source = prepared / "catalog.parquet"
    products.write_parquet(source)
    titles = products["product_title"].to_list()
    # These are predefined synthetic fixture labels, never ESCI judgments.
    development = pl.DataFrame(
        {
            "query_id": [1, 1, 2],
            "query": [titles[0], titles[0], titles[32]],
            "product_locale": ["us"] * 3,
            "product_id": [
                products["product_id"][0],
                products["product_id"][1],
                products["product_id"][32],
            ],
            "esci_label": ["E", "I", "I"],
            "group_sha256": ["fixture-g1", "fixture-g1", "fixture-g2"],
            "partition": ["development"] * 3,
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
        "scope": "synthetic_label_fixture_on_64_saved_pilot_vectors",
        "catalog_sha256": file_hash(source),
        "prepared_audit_sha256": file_hash(prepared / "audit.json"),
        "development_sha256": file_hash(prepared / "development.parquet"),
    }
    origin = {k: pins[k] for k in ("catalog_sha256", "prepared_audit_sha256")}
    resumed, direct = output / "resumed", output / "uninterrupted"
    child_code = None
    if process_interrupt:
        np.save(output / "replay.npy", vectors, allow_pickle=False)
        write_once(
            output / "child-control.json",
            {"origin": origin, "source_hash": report["vectors_sha256"]},
        )
        with (output / "child.log").open("wb") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--crash-child", str(output)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 20
                while not (output / "child-ready.json").exists():
                    if child.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Crash fixture child didn't reach its bounded checkpoint"
                        )
                    time.sleep(0.05)
                child.terminate()
                child_code = child.wait(timeout=5)
                if child_code >= 0:
                    raise AssertionError("Expected a signal-terminated fixture child")
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
    else:
        failed = ReplayEncoder(titles, vectors, report["vectors_sha256"], fail=True)
        try:
            encode_chunks(
                source, resumed, failed, origin, chunk_rows=32, wall_seconds=120, budget_root=root
            )
        except InterruptedError:
            pass
        else:
            raise AssertionError("Expected controlled interruption")
    first = resumed / "chunks/000000000000"
    before = {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in first.iterdir()}
    encoder = ReplayEncoder(titles, vectors, report["vectors_sha256"])
    resumed_report = encode_chunks(
        source, resumed, encoder, origin, chunk_rows=32, wall_seconds=120, budget_root=root
    )
    assert encoder.calls == 1
    assert before == {p.name: (file_hash(p), p.stat().st_mtime_ns) for p in first.iterdir()}
    direct_report = encode_chunks(
        source,
        direct,
        ReplayEncoder(titles, vectors, report["vectors_sha256"]),
        origin,
        chunk_rows=32,
        wall_seconds=120,
        budget_root=root,
    )
    assert resumed_report["chunks"] == direct_report["chunks"]
    for a, b in zip(completed_chunks(resumed), completed_chunks(direct), strict=True):
        assert a[0] == b[0] and a[1].equals(b[1])
        np.testing.assert_array_equal(a[2], b[2])
    a_bundle, b_bundle = output / "bundle-resumed", output / "bundle-direct"
    a_manifest = build_bundle(resumed, a_bundle, budget_root=root)
    b_manifest = build_bundle(direct, b_bundle, budget_root=root)
    assert a_manifest == b_manifest
    for name in BUNDLE_FILES | {"manifest.json", "COMPLETE.json"}:
        assert file_hash(a_bundle / name) == file_hash(b_bundle / name)
    index, mapping, metadata = load_bundle(a_bundle, encoder.recipe, expected_origin=origin)
    assert mapping.equals(products)
    query = encoder.encode([titles[0]])
    assert index.search(query, 64) == load_bundle(b_bundle, encoder.recipe)[0].search(query, 64)
    hits = index.search(query, 64)
    assert index.candidate_scores(query, [key for key, _ in hits]) == hits
    lexical = output / "fixture-lexical"
    build_lexical(products, lexical, origin)
    compared = evaluate_development(
        prepared, lexical, a_bundle, encoder, output / "fixture-evaluation", pins, resumed
    )
    result = {
        "schema_version": 1,
        "kind": (
            "64-row saved-pilot replay infrastructure fixture; "
            "no new model inference or real relevance evaluation"
        ),
        "products": 64,
        "dimensions": 384,
        "actual_model_inference_performed": False,
        "pilot_vectors_sha256": report["vectors_sha256"],
        "controlled_mid_chunk_interruption": True,
        "interruption_kind": "real child SIGTERM" if process_interrupt else "injected exception",
        "child_return_code": child_code,
        "unclosed_attempts_recovered": len(list((resumed / "attempts").glob("*.recovered.json"))),
        "committed_chunk_hashes_and_mtimes_unchanged": True,
        "committed_chunks_skipped": 1,
        "abandoned_partial_directories_preserved": len(list((resumed / "partials").iterdir())),
        "resumed_and_uninterrupted_chunk_ids_exact": True,
        "all_five_bundle_files_byte_identical": True,
        "dense_artifact_id": metadata["artifact_id"],
        "mapping_exact": True,
        "native_subset_scores_search_exact": True,
        "save_load_search_exact": True,
        "fixture_evaluator_methods": list(compared["A"]),
        "fixture_evaluator_queries": compared["A"]["dense"]["queries"],
        "fixture_labels_are_not_dataset_judgments": True,
        "real_development_or_test_judgments_loaded": False,
        "cumulative_resumed_job_seconds": accounted_seconds(resumed),
        "recipe_id": digest(encoder.recipe),
    }
    write_once(output / "report.json", result)
    write_once(output / "COMPLETE.json", {"report_sha256": file_hash(output / "report.json")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--crash-child"]:
        crash_child(Path(sys.argv[2]))
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-interrupt", action="store_true")
    args = parser.parse_args()
    run(args.pilot, args.output, args.process_interrupt)
