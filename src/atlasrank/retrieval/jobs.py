"""Bounded local build/evaluation jobs over approved prepared data only."""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from atlasrank.artifacts.lexical import InvalidIndex
from atlasrank.data.common import check_disk, check_memory, sha256, tree_bytes, write_json
from atlasrank.data.prepare import load_judgments
from atlasrank.evaluation.metrics import macro_average, score_labeled_candidates, score_query
from atlasrank.evaluation.slices import query_slices, summarize_slices
from atlasrank.evaluation.uncertainty import paired_interval
from atlasrank.retrieval.bm25 import SCORING, LexicalIndex, build, load


def configuration(path: Path, root: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text())
    if (
        spec.get("schema_version") != 1
        or spec.get("pipeline_id") != "bm25-title-lucene-v1"
        or any(spec.get(k) != v for k, v in SCORING.items())
        or spec.get("retrieval_k") != 100
        or spec.get("seed") != 42
        or spec.get("benchmark_queries") != 200
        or spec.get("benchmark_warmup") != 20
        or spec.get("benchmark_repetitions") != 3
    ):
        raise ValueError("Unsupported or unfrozen M3 configuration")
    for name in ("prepared_path", "index_path"):
        if not Path(spec[name]).resolve().is_relative_to(root.resolve()):
            raise ValueError("All inputs/indexes must stay inside the budgeted data root")
    prepared = Path(spec["prepared_path"])
    if sha256(prepared / "audit.json") != spec["prepared_audit_sha256"]:
        raise ValueError("Prepared audit differs from approved M2 snapshot")
    complete = json.loads((prepared / "COMPLETE.json").read_text())
    if (
        complete.get("schema_version") != 1
        or complete["audit_sha256"] != spec["prepared_audit_sha256"]
    ):
        raise ValueError("Incomplete M2 snapshot")
    if sha256(prepared / "catalog.parquet") != spec["catalog_sha256"]:
        raise ValueError("Catalog differs from approved M2 snapshot")
    return spec


def build_job(config: Path, root: Path, output: Path) -> Path:
    spec = configuration(config, root)
    prepared = Path(spec["prepared_path"])
    # Conservative disk reservation; every previous run remains within the root budget.
    check_disk(root, 4 * 1024**3)
    products = pl.read_parquet(
        prepared / "catalog.parquet",
        columns=[
            "product_locale",
            "product_id",
            "product_title",
        ],
    )
    report = build(
        products,
        output,
        {
            "prepared_audit_sha256": spec["prepared_audit_sha256"],
            "catalog_sha256": spec["catalog_sha256"],
            "config_sha256": sha256(config),
        },
    )
    report["bundle_path"] = str(output)
    report["bundle_bytes"] = tree_bytes(output)
    destination = root / "execution" / f"{output.name}-build.json"
    if destination.exists():
        raise FileExistsError("Build report exists; preserve previous execution history")
    write_json(destination, report)
    check_disk(root)
    return destination


def report_metrics(summary: dict[str, Any], track: str) -> dict[str, Any]:
    if track == "A":
        names = {"queries", "zero_idcg_queries", "ndcg_at_10", "mrr_at_10"}
    else:
        names = {
            "queries",
            "recall_queries",
            "recall_excluded_queries",
            "known_exact_recall_at_20",
            "known_exact_recall_at_100",
            "judgment_coverage_at_10",
            "judgment_coverage_at_100",
            "returned_at_10_total",
            "returned_at_100_total",
        }
    return {k: v for k, v in summary.items() if k in names}


def evaluate_job(config: Path, root: Path, output: Path) -> Path:
    spec = configuration(config, root)
    if output.exists():
        raise FileExistsError("Evaluation destination exists; preserve earlier runs")
    prepared = Path(spec["prepared_path"])
    if sha256(prepared / "development.parquet") != spec["development_sha256"]:
        raise ValueError("Development manifest differs from approved M2 snapshot")
    index = load(Path(spec["index_path"]))
    if index.metadata["origin"] != {
        "prepared_audit_sha256": spec["prepared_audit_sha256"],
        "catalog_sha256": spec["catalog_sha256"],
        "config_sha256": sha256(config),
    }:
        raise InvalidIndex("Index/source configuration lineage mismatch")
    build_report = json.loads(
        (root / "execution" / f"{Path(spec['index_path']).name}-build.json").read_text()
    )
    if index.probe_hashes() != build_report["probe_hashes_before_load"]:
        raise AssertionError("Actual-catalog save/load score parity failed")
    development = load_judgments(prepared, "development")
    groups = {
        int(key[0]): frame
        for key, frame in development.partition_by(
            "query_id", as_dict=True, maintain_order=True
        ).items()
    }
    a_results = {}
    b_results = {}
    memberships = {}
    group_keys = {}
    started = time.perf_counter()
    output.mkdir(parents=True)
    with (output / "per-query.jsonl").open("x") as per_query:
        for completed, query_id in enumerate(sorted(groups), 1):
            frame = groups[query_id]
            query = str(frame["query"][0])
            keys = [
                (str(locale), str(pid))
                for locale, pid in frame.select("product_locale", "product_id").iter_rows()
            ]
            judgments = dict(zip(keys, frame["esci_label"].to_list(), strict=True))
            query_started = time.perf_counter()
            scores = index.score_all(query)
            hits = index.hits_from_scores(scores, 100)
            retrieval_seconds = time.perf_counter() - query_started
            a = score_labeled_candidates(
                judgments, index.candidate_scores(scores, keys), index.catalog
            )
            b = score_query(
                judgments, [(hit.product_locale, hit.product_id) for hit in hits], index.catalog
            )
            a_results[query_id], b_results[query_id] = a, b
            group_keys[query_id] = str(frame["group_sha256"][0])
            titles = {key: index.titles[index.row_for_key[key]] for key in keys}
            memberships[query_id] = query_slices(query, judgments, titles)
            per_query.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "group_sha256": group_keys[query_id],
                        "A": report_metrics(asdict(a), "A"),
                        "B": report_metrics(asdict(b), "B"),
                        "returned_top100": [
                            {
                                "product_locale": hit.product_locale,
                                "product_id": hit.product_id,
                                "score": hit.score,
                            }
                            for hit in hits
                        ],
                        "retrieval_seconds": retrieval_seconds,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            if completed % 500 == 0:
                check_memory()
                check_disk(root)
                print(
                    json.dumps({"event": "development_progress", "queries": completed}), flush=True
                )
    report: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_id": spec["pipeline_id"],
        "bundle_id": index.bundle_id,
        "configuration_sha256": sha256(config),
        "development_sha256": spec["development_sha256"],
        "evaluation_contract": "v0.3",
        "partition": "development",
        "test_labels_accessed": False,
        "retrieval_k": 100,
        "positive_matches_only": True,
        "A": report_metrics(macro_average(list(a_results.values())), "A"),
        "B": report_metrics(macro_average(list(b_results.values())), "B"),
        "C": "not run: hybrid/retrieval-to-learned-reranking pipeline not implemented in M3",
        "elapsed_seconds": time.perf_counter() - started,
        "slices": {},
        "query_sampling_intervals": {},
        "incomplete_judgment_caveat": (
            "Unknown products remain unknown; known-exact recall is not complete catalog recall"
        ),
    }
    for track, results in (("A", a_results), ("B", b_results)):
        report["slices"][track] = {
            name: {value: report_metrics(metrics, track) for value, metrics in values.items()}
            for name, values in summarize_slices(results, memberships).items()
        }
    # A single-method mean interval uses group resampling; zeros are the algebraic
    # offset, not an invented competing model. There are no paired model claims.
    for track, field, results in (
        ("A", "ndcg_at_10", a_results),
        ("B", "known_exact_recall_at_20", b_results),
        ("B", "known_exact_recall_at_100", b_results),
    ):
        values = {q: getattr(result, field) for q, result in results.items()}
        if all(value is None for value in values.values()):
            report["query_sampling_intervals"][f"{track}.{field}"] = {
                "mean": None,
                "lower": None,
                "upper": None,
                "queries": 0,
                "groups": 0,
                "excluded_queries": len(values),
                "resamples": 2000,
                "seed": 42,
                "status": "undefined: no eligible queries",
            }
            continue
        zero_offset = {q: 0.0 if value is not None else None for q, value in values.items()}
        interval = asdict(paired_interval(zero_offset, values, group_keys))
        interval["mean"] = interval.pop("delta")
        report["query_sampling_intervals"][f"{track}.{field}"] = interval
    report["benchmark"] = benchmark(index, groups, spec)
    report["per_query_sha256"] = sha256(output / "per-query.jsonl")
    write_json(output / "report.json", report)
    write_json(
        output / "COMPLETE.json",
        {"schema_version": 1, "report_sha256": sha256(output / "report.json")},
    )
    check_memory()
    check_disk(root)
    return output


def benchmark(
    index: LexicalIndex, groups: dict[int, pl.DataFrame], spec: dict[str, Any]
) -> dict[str, Any]:
    import hashlib

    workload = sorted(groups, key=lambda q: hashlib.sha256(f"42|{q}".encode()).hexdigest())[:200]
    queries = [str(groups[q]["query"][0]) for q in workload]
    for query in queries[:20]:
        index.search(query, 100)
    runs = []
    for repetition in range(3):
        times = []
        for query in queries:
            started = time.perf_counter_ns()
            index.search(query, 100)
            times.append((time.perf_counter_ns() - started) / 1e6)
        runs.append(
            {
                "repetition": repetition + 1,
                "queries": len(times),
                "latency_ms": {
                    p: float(np.percentile(times, q))
                    for p, q in (("p50", 50), ("p95", 95), ("p99", 99))
                },
                "per_query_latency_ms": times,
            }
        )
    return {
        "schema_version": 1,
        "kind": "basic offline sequential latency, not M8 load testing",
        "query_ids": workload,
        "query_ids_sha256": hashlib.sha256(json.dumps(workload).encode()).hexdigest(),
        "warmup_queries": min(20, len(queries)),
        "top_k": spec["retrieval_k"],
        "runs": runs,
        "includes": (
            "analyze, score all catalog documents, deterministic positive top-100 selection"
        ),
        "excludes": "HTTP, JSON, queueing, cold load, model encoding",
    }
