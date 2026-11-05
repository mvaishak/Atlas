"""Verify saved pilot vectors against NumPy and native FAISS I/O, without encoding."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import polars as pl

from atlasrank.data.common import check_disk, sha256, write_json
from atlasrank.evaluation.vector_checks import compare_reference_ranking
from atlasrank.retrieval.dense import DenseIndex


def check(pilot: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Preserve previous vector checks")
    root = Path("data/esci")
    if not all(p.resolve().is_relative_to(root.resolve()) for p in (pilot, output)):
        raise ValueError("Pilot artifacts must remain inside the budgeted data root")
    report = json.loads((pilot / "report.json").read_text())
    if (
        sha256(pilot / "report.json")
        != json.loads((pilot / "COMPLETE.json").read_text())["report_sha256"]
    ):
        raise ValueError("Pilot completion checksum differs")
    for name, field in (("sample.parquet", "sample_sha256"), ("vectors.npy", "vectors_sha256")):
        if sha256(pilot / name) != report[field]:
            raise ValueError("Pilot component checksum differs")
    sample = pl.read_parquet(pilot / "sample.parquet")
    keys = list(sample.select("product_locale", "product_id").iter_rows())
    vectors = np.load(pilot / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if vectors.shape != (4096, 384) or len(keys) != 4096:
        raise ValueError("Expected the approved 4096-title calibration")
    check_disk(root, 30 * 1024**2)
    index = DenseIndex(keys, vectors)
    row_for_key = {key: row for row, key in enumerate(keys)}
    comparisons = []
    for row in (0, 2048, 4095):
        query = np.asarray(vectors[row : row + 1])
        hits = index.search(query, 100)
        reference = (np.asarray(vectors) @ query.T).ravel()
        compared = compare_reference_ranking(
            reference,
            [row_for_key[key] for key, _ in hits],
            [score for _, score in hits],
            100,
        )
        for _ in range(3):
            if index.search(query, 100) != hits:
                raise AssertionError("Repeated reference FAISS search differs")
        comparisons.append({"sample_query_row": row, **compared, "repeat_search_exact": True})
    output.mkdir(parents=True)
    before = index.search(vectors[:1], 100)
    faiss.write_index(index.index, str(output / "fixture.faiss"))
    restored = faiss.read_index(str(output / "fixture.faiss"))
    if not isinstance(restored, faiss.IndexFlatIP):
        raise AssertionError("Unexpected saved index type")
    np.testing.assert_array_equal(restored.reconstruct_n(0, len(keys)), vectors)
    index.index = restored
    if index.search(vectors[:1], 100) != before:
        raise AssertionError("Save/load search differs")
    result = {
        "schema_version": 1,
        "kind": "Pilot vectors only; not full-catalog retrieval or relevance evaluation",
        "products": len(keys),
        "dimensions": 384,
        "sample_sha256": report["sample_sha256"],
        "vectors_sha256": report["vectors_sha256"],
        "index_sha256": sha256(output / "fixture.faiss"),
        "index_bytes": (output / "fixture.faiss").stat().st_size,
        "max_unit_norm_error": float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1))),
        "brute_force_comparisons": comparisons,
        "saved_vector_rows_exact": True,
        "save_load_search_exact": True,
        "judgments_loaded": False,
    }
    write_json(output / "report.json", result)
    write_json(output / "COMPLETE.json", {"report_sha256": sha256(output / "report.json")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.pilot, args.output)
