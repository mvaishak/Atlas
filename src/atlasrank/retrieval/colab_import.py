"""Bounded local review of returned pilot chunks; rebuild native FAISS on the Mac."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from atlasrank.artifacts.dense import product_keys, write_once
from atlasrank.evaluation.cuda_checks import compare_cuda_vectors
from atlasrank.evaluation.vector_checks import compare_reference_ranking
from atlasrank.retrieval.chunks import completed_chunks
from atlasrank.retrieval.colab_pilot import restore_job
from atlasrank.retrieval.colab_transport import sha
from atlasrank.retrieval.dense import DenseIndex
from atlasrank.retrieval.dense_bundle import build_bundle, load_bundle


def review_pilot(store: Path, inputs: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve earlier imports")
    transfer = json.loads((inputs / "TRANSFER.json").read_text())
    if transfer["kind"] != "m4_cpu_exact_label_free_pilot" or transfer["rows"] != 4096:
        raise ValueError("This local command accepts only the bounded pilot")
    for name, pin in transfer["files"].items():
        target = inputs / name
        if not target.resolve().is_relative_to(inputs.resolve()) or sha(target) != pin:
            raise ValueError("Local reference transfer changed")
    if any(p.is_symlink() for p in store.rglob("*")):
        raise ValueError("Symlink imports unsupported")
    if sum(p.stat().st_size for p in store.rglob("*") if p.is_file()) > 1024**3:
        raise ValueError("Pilot return exceeds persistent budget")
    output.mkdir(parents=True)
    staging = output / "staging"
    staging.mkdir()
    job = output / "job"
    restore_job(store, job, staging, 256 * 1024**2)
    metadata = json.loads((job / "JOB.json").read_text())
    expected = pl.read_parquet(inputs / "pilot/products.parquet")
    if metadata["rows"] != 4096 or metadata["origin"] != transfer["origin"]:
        raise ValueError("Returned job source/origin/count differs")
    records = list(completed_chunks(job))
    products = pl.concat([p for _, p, _ in records])
    if not products.equals(expected):
        raise ValueError("Returned identities/titles/order differ from CPU pilot")
    vectors = np.concatenate([v for _, _, v in records])
    cpu = np.load(inputs / "pilot/cpu-vectors.npy", allow_pickle=False)
    numerical = compare_cuda_vectors(cpu, vectors)
    build_bundle(job, output / "bundle", budget_root=output)
    index, mapping, bundle = load_bundle(
        output / "bundle", metadata["recipe"], expected_origin=transfer["origin"]
    )
    if not mapping.equals(expected):
        raise ValueError("Local FAISS bundle mapping changed")
    keys = product_keys(products)
    reference = DenseIndex(keys, cpu)
    ranking = []
    for i in range(16):
        # Same CPU query vector on both indexes isolates product-vector differences.
        query = cpu[i : i + 1]
        a = reference.search(query, 100)
        b = index.search(query, 100)
        native_cpu = np.asarray(
            [s for _, s in reference.candidate_scores(query, keys)], dtype=np.float32
        )
        native_cuda = np.asarray(
            [s for _, s in index.candidate_scores(query, keys)], dtype=np.float32
        )
        np.testing.assert_allclose(native_cuda, native_cpu, rtol=0, atol=1e-4)
        ranking.append(
            {
                "probe_row": i,
                "top100_membership_exact": {k for k, _ in a} == {k for k, _ in b},
                "top100_order_exact": [k for k, _ in a] == [k for k, _ in b],
                "same_index_numpy_diagnostic": compare_reference_ranking(
                    vectors @ query[0], [index.row_for_key[k] for k, _ in b], [s for _, s in b], 100
                ),
                "cross_device_score_max_error": float(np.max(np.abs(native_cuda - native_cpu))),
            }
        )
    report = {
        "schema_version": 1,
        "rows": 4096,
        "numerical": numerical,
        "local_faiss_checks": ranking,
        "bundle_id": bundle["artifact_id"],
        "labels_loaded": False,
        "development_quality_run": False,
        "full_run_approved": False,
    }
    write_once(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("store", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    review_pilot(args.store, args.inputs, args.output)


if __name__ == "__main__":
    main()
