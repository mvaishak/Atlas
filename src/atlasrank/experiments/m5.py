"""Thin M5 reproduction CLI. Run each neural/native stage in its own process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from atlasrank.artifacts.dense import file_hash, product_keys, write_once
from atlasrank.data.prepare import load_judgments
from atlasrank.evaluation.m5 import (
    compare_pairs,
    evaluate_rankings,
    export_bm25,
    read_rankings,
    track_a,
    validate_development,
)
from atlasrank.retrieval.m5_encoder import M5Encoder, verify_file, verify_model, verify_recipe


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def development(prepared: Path, pins: dict[str, Any]) -> pl.DataFrame:
    verify_file(prepared / "development.parquet", pins["development_sha256"])
    frame = load_judgments(prepared, "development")
    validate_development(frame)
    return frame


def paired_inputs(
    m4: Path, compact: Path, frame: pl.DataFrame, pins: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    validate_development(frame)
    verify_file(m4, pins["m4_per_query_sha256"])
    verify_file(compact, pins["m5_compact_sha256"])
    expected = dict(frame.select("query_id", "group_sha256").unique().iter_rows())
    old = {}
    with m4.open() as source:
        for line in source:
            row = json.loads(line)
            qid = int(row["query_id"])
            if qid in old or expected.get(qid) != row["group_sha256"]:
                raise ValueError("M4 query/group identities differ")
            old[qid] = row
    new = load_json(compact)
    if (
        new["partition"] != "development"
        or new["query_count"] != len(expected)
        or new["model_weights_sha256"] != pins["model_weights_sha256"]
        or {int(q) for q in new["metrics"]} != expected.keys()
        or old.keys() != expected.keys()
    ):
        raise ValueError("M5 paired evidence provenance/query IDs differ")
    return {
        f"{pipeline}.{metric}": [
            {
                "query_id": q,
                "group": expected[q],
                "reference": old[q]["metrics"]["B"][pipeline][metric],
                "variant": new["metrics"][str(q)][pipeline][metric],
            }
            for q in sorted(expected)
        ]
        for pipeline in ["dense", "hybrid"]
        for metric in ["known_exact_recall_at_20", "known_exact_recall_at_100"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m5-reproduction.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("verify-inputs")
    audit.add_argument("--archive", type=Path, required=True)
    audit.add_argument("--model", type=Path, required=True)
    audit.add_argument("--catalog", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    pairs = sub.add_parser("paired")
    pairs.add_argument("--inputs", type=Path, required=True)
    pairs.add_argument("--output", type=Path, required=True)
    collect = sub.add_parser("collect-pairs")
    collect.add_argument("--prepared", type=Path, required=True)
    collect.add_argument("--m4", type=Path, required=True)
    collect.add_argument("--compact", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    for name in ["track-a", "encode-queries", "export-bm25", "track-b"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--prepared", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if name in {"track-a", "encode-queries"}:
            cmd.add_argument("--model", type=Path, required=True)
            cmd.add_argument("--device", default="cpu")
        if name == "track-a":
            cmd.add_argument("--catalog", type=Path, required=True)
        if name == "export-bm25":
            cmd.add_argument("--lexical", type=Path, required=True)
        if name == "track-b":
            cmd.add_argument("--catalog", type=Path, required=True)
            cmd.add_argument("--chunks", type=Path, required=True)
            cmd.add_argument("--index", type=Path, required=True)
            cmd.add_argument(
                "--index-format",
                choices=["historical", "rebuilt-historical", "canonical"],
                default="historical",
            )
            cmd.add_argument("--queries", type=Path, required=True)
            cmd.add_argument("--bm25", type=Path, required=True)
    for name in ["encode-catalog", "build-index", "verify-index", "build-canonical-index"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--catalog", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if name == "encode-catalog":
            cmd.add_argument("--model", type=Path, required=True)
            cmd.add_argument("--device", required=True)
            cmd.add_argument("--wall-seconds", type=float, required=True)
            cmd.add_argument("--persistent-store", type=Path)
        else:
            cmd.add_argument("--chunks", type=Path, required=True)
    args = parser.parse_args()
    pins = load_json(args.config)
    verify_file(Path("docs/EVALUATION.md"), pins["evaluation_contract_sha256"])
    command = args.command
    result: Any
    if command == "verify-inputs":
        verify_file(args.archive, pins["input_zip_sha256"])
        verify_model(args.model, pins["model_files"])
        verify_file(args.catalog, pins["catalog_sha256"])
        import pyarrow.parquet as pq

        if pq.ParquetFile(args.catalog).metadata.num_rows != pins["catalog_rows"]:
            raise ValueError("Canonical count differs")
        result = {
            "archive_sha256": file_hash(args.archive),
            "catalog_sha256": file_hash(args.catalog),
            "model_inventory_verified": True,
        }
        write_once(args.output, result)
    elif command == "collect-pairs":
        result = paired_inputs(args.m4, args.compact, development(args.prepared, pins), pins)
        write_once(args.output, result)
    elif command == "paired":
        result = {name: compare_pairs(rows) for name, rows in load_json(args.inputs).items()}
        write_once(args.output, result)
    elif command in {"encode-catalog", "build-canonical-index"}:
        verify_file(args.catalog, pins["catalog_sha256"])
        from atlasrank.retrieval.colab_full import catalog_contract, verify_coverage

        contract = catalog_contract(args.catalog, pins["catalog_rows"], pins["chunk_rows"])
        if command == "encode-catalog":
            from atlasrank.retrieval.chunks import encode_chunks

            encoder = M5Encoder(args.model, pins, device=args.device, catalog=True)
            origin = {"experiment": "m5-consolidated-v1", "weights": pins["model_weights_sha256"]}
            if args.persistent_store:
                from atlasrank.retrieval.colab_full import encode_persisted

                result = encode_persisted(
                    args.catalog,
                    args.output,
                    encoder,
                    origin,
                    args.persistent_store,
                    chunk_rows=pins["chunk_rows"],
                    wall_seconds=args.wall_seconds,
                )
            else:
                result = encode_chunks(
                    args.catalog,
                    args.output,
                    encoder,
                    origin,
                    chunk_rows=pins["chunk_rows"],
                    wall_seconds=args.wall_seconds,
                )

        else:
            # Native-only process. Fully validate canonical rows before index construction.
            from atlasrank.retrieval.dense_bundle import build_bundle, load_bundle

            job = load_json(args.chunks / "JOB.json")
            verify_recipe(job["recipe"], pins, catalog=True)
            verify_coverage(args.chunks, args.catalog, contract, job["recipe"], job["origin"])
            result = build_bundle(args.chunks, args.output)
            load_bundle(args.output, job["recipe"], expected_origin=job["origin"])
    elif command in {"build-index", "verify-index"}:
        from atlasrank.retrieval.m5_artifacts import (
            build_historical_index,
            validate_historical_index,
        )

        if command == "build-index":
            result = build_historical_index(args.catalog, args.chunks, args.output, pins)
        else:
            index, _ = validate_historical_index(
                args.output,
                {**pins, **pins["historical_index"]},
                source=args.catalog,
                chunks=args.chunks,
            )
            result = {"verified_vectors": index.ntotal}
    else:
        frame = development(args.prepared, pins)
        if command == "export-bm25":
            from atlasrank.retrieval.bm25 import load

            verify_file(args.lexical / "manifest.json", pins["lexical_manifest_sha256"])
            result = export_bm25(frame, load(args.lexical), args.output)
        elif command in {"track-a", "encode-queries"}:
            encoder = M5Encoder(args.model, pins, device=args.device)
            if command == "track-a":
                verify_file(args.catalog, pins["catalog_sha256"])
                rows, summary = track_a(frame, pl.read_parquet(args.catalog), encoder)
                result = {
                    "track": "A",
                    "summary": summary,
                    "per_query": rows,
                    "recipe": encoder.recipe,
                }
                write_once(args.output, result)
            else:
                if args.output.exists():
                    raise FileExistsError("Preserve encoded queries")
                args.output.mkdir(parents=True)
                queries = (
                    frame.select("query_id", "query", "group_sha256").unique().sort("query_id")
                )
                texts = queries["query"].to_list()
                vectors = np.concatenate(
                    [encoder.encode(texts[i : i + 32]) for i in range(0, len(texts), 32)]
                )
                np.save(args.output / "vectors.npy", vectors, allow_pickle=False)
                queries.select("query_id", "group_sha256").write_parquet(
                    args.output / "queries.parquet"
                )
                result = {
                    "recipe": encoder.recipe,
                    "development_sha256": pins["development_sha256"],
                    "files": {
                        n: file_hash(args.output / n) for n in ["vectors.npy", "queries.parquet"]
                    },
                }
                write_once(args.output / "manifest.json", result)
        else:
            result = run_track_b(args, pins, frame)
    print(
        json.dumps(
            result
            if command not in {"collect-pairs", "track-a", "track-b"}
            else {"output": str(args.output), "summary": result.get("summary", list(result))},
            indent=2,
            allow_nan=False,
        )
    )


def run_track_b(
    args: argparse.Namespace, pins: dict[str, Any], frame: pl.DataFrame
) -> dict[str, Any]:
    """Historical batched FAISS search, followed by existing metrics and RRF-60."""
    import faiss

    from atlasrank.retrieval.m5_artifacts import validate_historical_index, verify_bundle_job

    if args.output.exists():
        raise FileExistsError("Preserve evaluations")
    if args.index_format in {"historical", "rebuilt-historical"}:
        index_pins = (
            {**pins, **pins["historical_index"]} if args.index_format == "historical" else pins
        )
        index, ids = validate_historical_index(
            args.index,
            index_pins,
            source=args.catalog,
            chunks=args.chunks,
        )
        index_manifest = load_json(args.index / "INDEX_MANIFEST.json")
    else:
        from atlasrank.retrieval.colab_full import catalog_contract, verify_coverage
        from atlasrank.retrieval.dense_bundle import load_bundle

        job = load_json(args.chunks / "JOB.json")
        verify_recipe(job["recipe"], pins, catalog=True)
        verify_file(args.catalog, pins["catalog_sha256"])
        contract = catalog_contract(args.catalog, pins["catalog_rows"], pins["chunk_rows"])
        coverage = verify_coverage(
            args.chunks, args.catalog, contract, job["recipe"], job["origin"]
        )
        adapter, products, index_manifest = load_bundle(
            args.index, job["recipe"], expected_origin=job["origin"]
        )
        verify_bundle_job(index_manifest, job, coverage)
        ids = product_keys(products)
        index = adapter.index
    manifest = load_json(args.queries / "manifest.json")
    verify_recipe(manifest["recipe"], pins)
    if (
        manifest["development_sha256"] != pins["development_sha256"]
        or manifest["recipe"]["model_files"] != pins["model_files"]
        or manifest["recipe"]["kind"] != "m5-fixed-query-v1"
        or set(manifest["files"]) != {"vectors.npy", "queries.parquet"}
    ):
        raise ValueError("Query provenance differs")
    for name, sha in manifest["files"].items():
        verify_file(args.queries / name, sha)
    queries = pl.read_parquet(args.queries / "queries.parquet")
    expected = frame.select("query_id", "group_sha256").unique().sort("query_id")
    vectors = np.load(args.queries / "vectors.npy", allow_pickle=False)
    if (
        not queries.equals(expected)
        or vectors.dtype != np.float32
        or vectors.shape != (queries.height, pins["dimensions"])
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Query vector alignment differs")
    faiss.omp_set_num_threads(4)
    dense = {}
    qids = queries["query_id"].to_list()
    for start in range(0, len(qids), 16):
        _, indices = index.search(
            np.ascontiguousarray(vectors[start : start + 16]), min(100, len(ids))
        )
        for qid, ranked in zip(qids[start : start + 16], indices, strict=True):
            if (ranked < 0).any():
                raise ValueError("Incomplete native ranking")
            dense[int(qid)] = [ids[int(row)] for row in ranked]
    verify_file(args.bm25, pins["bm25_export_sha256"])
    per_query, summary = evaluate_rankings(frame, read_rankings(args.bm25), dense, frozenset(ids))
    result = {
        "track": "B",
        "retrieval": "historical-faiss-batch16-no-tie-refinement",
        "summary": summary,
        "per_query": per_query,
        "dense_rankings": dense,
        "bm25_sha256": file_hash(args.bm25),
        "index_manifest": index_manifest,
        "index_format": args.index_format,
        "query_manifest": manifest,
    }
    write_once(args.output, result)
    return result


if __name__ == "__main__":
    main()
