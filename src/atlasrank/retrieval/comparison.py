"""Development-only A/B comparisons; no judged-candidate injection into retrieval."""

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from atlasrank.artifacts.dense import InvalidDense, digest, file_hash, write_once
from atlasrank.data.common import check_disk
from atlasrank.data.prepare import load_judgments
from atlasrank.evaluation.metrics import (
    QueryMetrics,
    macro_average,
    rank_scores,
    score_labeled_candidates,
    score_query,
)
from atlasrank.evaluation.slices import query_slices, summarize_slices
from atlasrank.evaluation.uncertainty import paired_interval
from atlasrank.retrieval.bm25 import LexicalIndex
from atlasrank.retrieval.bm25 import load as load_lexical
from atlasrank.retrieval.chunks import Encoder, account_work
from atlasrank.retrieval.dense import DenseIndex
from atlasrank.retrieval.dense_bundle import load_bundle
from atlasrank.retrieval.fusion import reciprocal_rank_fusion
from atlasrank.retrieval.jobs import report_metrics

METHODS = ("bm25", "dense", "hybrid")


def verify_bm25_reference(comparison: Path, reference: Path) -> None:
    """Require exact M3 development metrics, ranks and scores, excluding timers."""
    prior = {r["query_id"]: r for r in (json.loads(line) for line in reference.open())}
    current = {
        r["query_id"]: r
        for r in (json.loads(line) for line in (comparison / "per-query.jsonl").open())
    }
    if prior.keys() != current.keys():
        raise InvalidDense("M3 reference query membership differs")
    for query_id, row in current.items():
        old = prior[query_id]
        expected_scores = [
            ([h["product_locale"], h["product_id"]], h["score"]) for h in old["returned_top100"]
        ]
        scores = [(key, score) for key, score in row["B_scores"]["bm25"]]
        if (
            row["group_sha256"] != old["group_sha256"]
            or row["metrics"]["A"]["bm25"] != old["A"]
            or row["metrics"]["B"]["bm25"] != old["B"]
            or scores != expected_scores
        ):
            raise InvalidDense("BM25 differs from the frozen M3 development reference")


def compare_development(
    development: pl.DataFrame,
    lexical: LexicalIndex,
    dense: DenseIndex,
    encoder: Encoder,
    output: Path,
    provenance: dict[str, Any],
    checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve previous comparisons")
    required = {
        "query_id",
        "query",
        "group_sha256",
        "partition",
        "product_locale",
        "product_id",
        "esci_label",
    }
    if not required.issubset(development.columns) or any(
        development[name].null_count() for name in required
    ):
        raise ValueError("Missing/null development fields")
    development = development.sort("query_id", "product_locale", "product_id")
    if tuple(lexical.keys) != dense.keys:
        raise InvalidDense("Lexical/dense full catalog identities/order differ")
    if not development.height or set(development["partition"].to_list()) != {"development"}:
        raise ValueError("Only a nonempty development partition is admissible")
    if (
        development.unique(subset=["product_locale", "query_id", "product_id"]).height
        != development.height
    ):
        raise ValueError("Duplicate development query/product pairs")
    if provenance["recipe_id"] != digest(encoder.recipe):
        raise InvalidDense("Evaluation query encoder differs from artifact recipe")
    results: dict[str, dict[str, dict[int, QueryMetrics]]] = {
        track: {method: {} for method in METHODS} for track in ("A", "B")
    }
    groups: dict[int, str] = {}
    memberships = {}
    output.mkdir(parents=True)
    with (output / "per-query.jsonl").open("x") as saved:
        for key, frame in development.partition_by(
            "query_id", as_dict=True, maintain_order=True
        ).items():
            if checkpoint is not None:
                checkpoint()
            query_id = int(key[0])
            if frame["query"].n_unique() != 1 or frame["group_sha256"].n_unique() != 1:
                raise ValueError("Query ID maps to inconsistent text/group")
            query = str(frame["query"][0])
            groups[query_id] = str(frame["group_sha256"][0])
            if not groups[query_id]:
                raise ValueError("Missing normalized-query group")
            keys = [
                (str(locale), str(pid))
                for locale, pid in frame.select("product_locale", "product_id").iter_rows()
            ]
            judgments = dict(zip(keys, frame["esci_label"].to_list(), strict=True))
            lexical_scores = lexical.score_all(query)
            query_vector = encoder.encode([query])
            # A ranks every judged candidate, even those absent from B's top 100.
            candidate_scores = {
                "bm25": lexical.candidate_scores(lexical_scores, keys),
                "dense": dense.candidate_scores(query_vector, keys),
            }
            a_rankings = {
                method: rank_scores(scores) for method, scores in candidate_scores.items()
            }
            candidate_scores["hybrid"] = reciprocal_rank_fusion(
                [a_rankings["bm25"], a_rankings["dense"]], len(keys)
            )
            a_rankings["hybrid"] = rank_scores(candidate_scores["hybrid"])
            # B sees only independent catalog searches; labels never select candidates.
            lexical_hits = lexical.hits_from_scores(lexical_scores, 100)
            b_scores = {
                "bm25": [((h.product_locale, h.product_id), h.score) for h in lexical_hits],
                "dense": dense.search(query_vector, 100),
            }
            b_rankings = {method: [key for key, _ in scores] for method, scores in b_scores.items()}
            b_scores["hybrid"] = reciprocal_rank_fusion(
                [b_rankings["bm25"], b_rankings["dense"]], 100
            )
            b_rankings["hybrid"] = [key for key, _ in b_scores["hybrid"]]
            for method in METHODS:
                results["A"][method][query_id] = score_labeled_candidates(
                    judgments, candidate_scores[method], lexical.catalog
                )
                results["B"][method][query_id] = score_query(
                    judgments, b_rankings[method], lexical.catalog
                )
            titles = {key: lexical.titles[lexical.row_for_key[key]] for key in keys}
            memberships[query_id] = query_slices(query, judgments, titles)
            saved.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "group_sha256": groups[query_id],
                        "A_rankings": a_rankings,
                        "B_rankings": b_rankings,
                        "A_scores": candidate_scores,
                        "B_scores": b_scores,
                        "metrics": {
                            track: {
                                method: report_metrics(
                                    asdict(results[track][method][query_id]), track
                                )
                                for method in METHODS
                            }
                            for track in ("A", "B")
                        },
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    report: dict[str, Any] = {
        "schema_version": 1,
        "partition": "development",
        "evaluation_contract": "v0.3",
        "provenance": provenance,
        "retriever_k": 100,
        "rrf_constant": 60,
        "A": {},
        "B": {},
        "slices": {},
        "paired_intervals": {},
        "C": "Not run; hybrid candidate lists are a pre-ranker reference only",
        "incomplete_judgments": (
            "Unknown products remain unknown; known-exact recall is not complete catalog recall"
        ),
        "test_judgments_accessed": False,
        "per_query_sha256": file_hash(output / "per-query.jsonl"),
    }
    for track in ("A", "B"):
        report[track] = {
            method: report_metrics(macro_average(list(results[track][method].values())), track)
            for method in METHODS
        }
        report["slices"][track] = {
            method: {
                name: {value: report_metrics(metrics, track) for value, metrics in values.items()}
                for name, values in summarize_slices(results[track][method], memberships).items()
            }
            for method in METHODS
        }
    for track, field in (
        ("A", "ndcg_at_10"),
        ("B", "known_exact_recall_at_20"),
        ("B", "known_exact_recall_at_100"),
    ):
        for reference, variant in (("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid")):
            a = {q: getattr(r, field) for q, r in results[track][reference].items()}
            b = {q: getattr(r, field) for q, r in results[track][variant].items()}
            name = f"{track}.{field}.{variant}_minus_{reference}"
            report["paired_intervals"][name] = (
                asdict(paired_interval(a, b, groups))
                if any(value is not None for value in a.values())
                else {
                    "delta": None,
                    "lower": None,
                    "upper": None,
                    "queries": 0,
                    "groups": 0,
                    "excluded_queries": len(a),
                    "status": "undefined: no eligible known-E queries",
                    "resamples": 2000,
                    "seed": 42,
                }
            )
    write_once(output / "report.json", report)
    write_once(output / "COMPLETE.json", {"report_sha256": file_hash(output / "report.json")})
    return report


def evaluate_development(
    prepared: Path,
    lexical_directory: Path,
    dense_directory: Path,
    encoder: Encoder,
    output: Path,
    expected_source: dict[str, Any],
    job_directory: Path,
    checkpoint: Callable[[], None] | None = None,
    *,
    product_recipe: dict[str, Any] | None = None,
    accounting_directory: Path | None = None,
) -> dict[str, Any]:
    """Guard all source pins before opening development labels. No split selector."""
    with account_work(accounting_directory or job_directory, "development_evaluation") as tick:

        def guarded_tick() -> None:
            tick()
            if checkpoint is not None:
                checkpoint()

        guarded_tick()
        for filename, pin in (
            ("audit.json", "prepared_audit_sha256"),
            ("catalog.parquet", "catalog_sha256"),
            ("development.parquet", "development_sha256"),
        ):
            if file_hash(prepared / filename) != expected_source[pin]:
                raise InvalidDense("Evaluation source differs from pinned preparation")
        job = json.loads((job_directory / "JOB.json").read_text())
        if product_recipe is not None:
            # Explicit imported-CUDA products / local CPU queries. Never disguise
            # the query encoder as the product backend merely to bypass validation.
            fields = (
                "repo",
                "revision",
                "model_files_sha256",
                "dimensions",
                "max_length",
                "preprocessing",
                "padding",
                "pooling",
                "dtype",
                "attention",
            )
            if any(
                k not in product_recipe
                or k not in encoder.recipe
                or product_recipe[k] != encoder.recipe[k]
                for k in fields
            ):
                raise InvalidDense("Product/query encoder semantics differ")
        dense, products, manifest = load_bundle(
            dense_directory, product_recipe or encoder.recipe, expected_origin=job["origin"]
        )
        if manifest["job_id"] != job["job_id"] or any(
            manifest["origin"].get(k) != expected_source[k]
            for k in ("catalog_sha256", "prepared_audit_sha256")
        ):
            raise InvalidDense("Dense source/ledger lineage differs")
        lexical = load_lexical(lexical_directory)
        if any(
            lexical.metadata["origin"].get(k) != expected_source[k]
            for k in ("catalog_sha256", "prepared_audit_sha256")
        ):
            raise InvalidDense("Lexical preparation differs")
        if products["product_title"].to_list() != lexical.titles or dense.keys != tuple(
            lexical.keys
        ):
            raise InvalidDense("Lexical/dense catalog content differs")
        guarded_tick()
        development = load_judgments(prepared, "development")
        report = compare_development(
            development,
            lexical,
            dense,
            encoder,
            output,
            {
                "recipe_id": digest(encoder.recipe),
                "query_recipe": encoder.recipe,
                "product_recipe_id": manifest["recipe_id"],
                "product_recipe": manifest["recipe"],
                "dense_artifact_id": manifest["artifact_id"],
                "lexical_artifact_id": lexical.bundle_id,
                "source": expected_source,
            },
            checkpoint=guarded_tick,
        )
        guarded_tick()
        check_disk(output.parent)
        return report
