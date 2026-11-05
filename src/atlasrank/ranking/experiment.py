"""Single fixed M4 ranker experiment, gated on real artifacts and exact HTTP parity.

Linux worker only. The parent runner enforces wall/RSS/disk limits across native calls.
"""

import gc
import importlib.metadata
import json
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
from fastapi.testclient import TestClient

from atlasrank.artifacts.dense import digest, file_hash, seal, write_once
from atlasrank.data.common import peak_rss_bytes
from atlasrank.data.prepare import load_judgments
from atlasrank.evaluation.metrics import (
    assert_candidate_membership,
    macro_average,
    score_labeled_candidates,
    score_query,
)
from atlasrank.evaluation.ranking import offline_predict
from atlasrank.evaluation.uncertainty import paired_interval
from atlasrank.features.ranking import (
    FEATURE_NAMES,
    SCHEMA,
    SCHEMA_ID,
    DenseBinding,
    FeatureContext,
    build_features,
    model_identity,
)
from atlasrank.ranking.experiment_contract import (
    GRADES,
    fixed_configuration,
    require_fit_budget,
    validate_baseline,
    validate_groups,
)
from atlasrank.ranking.learned import FittedPredictor, XGBoostRanker
from atlasrank.ranking.prediction import DeterministicRanker, assert_exact_parity
from atlasrank.ranking.xgboost_boundary import ZeroUpdateModel
from atlasrank.retrieval.bm25 import load as load_lexical
from atlasrank.retrieval.dense_bundle import load_bundle
from atlasrank.retrieval.encoder import PinnedEncoder, configuration
from atlasrank.retrieval.fusion import reciprocal_rank_fusion
from atlasrank.serving.app import create_app
from atlasrank.serving.ranking import RankingRuntime


def read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def source_snapshot(source: Path) -> dict[str, Any]:
    manifest = read_json(source / "SOURCE_MANIFEST.json")
    for name, pin in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe source manifest path")
        path = source / name
        if path.is_symlink() or file_hash(path) != pin["sha256"]:
            raise ValueError(f"Executed source differs: {name}")
    if not Path(__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("Worker imported outside the verified source snapshot")
    return {"manifest_sha256": file_hash(source / "SOURCE_MANIFEST.json"), **manifest}


class Guard:
    def __init__(self, output: Path, wall_seconds: int, rss_limit: int) -> None:
        self.started, self.wall_seconds, self.rss_limit = time.monotonic(), wall_seconds, rss_limit
        self.output = output

    def check(self) -> None:
        if time.monotonic() - self.started >= self.wall_seconds:
            raise RuntimeError("Execution wall ceiling reached")
        if peak_rss_bytes() > self.rss_limit or psutil.virtual_memory().available < 768 * 1024**2:
            raise RuntimeError("Measured memory headroom exhausted")

    def event(self, stage: str, **fields: Any) -> None:
        self.check()
        row = {
            "stage": stage,
            "elapsed_seconds": time.monotonic() - self.started,
            "peak_rss_bytes": peak_rss_bytes(),
            **fields,
        }
        with (self.output / "progress.jsonl").open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


def verified_inputs(inputs: Path, source: Path) -> dict[str, Any]:
    pins = read_json(inputs / "INPUT_MANIFEST.json")
    for name, pin in pins.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe input path")
        path = inputs / name
        if (
            path.is_symlink()
            or path.stat().st_size != pin["bytes"]
            or file_hash(path) != pin["sha256"]
        ):
            raise ValueError(f"Input checksum mismatch: {name}")
    frozen = read_json(source / "docs/reviews/M4.cpu-run.validation.json")
    expected = frozen["source_and_contract"]
    audit = read_json(inputs / "prepared/audit.json")
    for name, sha in (
        ("prepared/audit.json", expected["audit_sha256"]),
        ("prepared/catalog.parquet", expected["catalog_sha256"]),
        ("prepared/development.parquet", expected["development_sha256"]),
        (
            "prepared/train.parquet",
            "0c7c2eb055d5a03d785ccf93bfb77713cc619bb0dcc3665473c1b6d2f37e62f9",
        ),
    ):
        if pins[name]["sha256"] != sha:
            raise ValueError(f"Original source pin differs: {name}")
    if file_hash(source / "docs/EVALUATION.md") != expected["evaluation_contract_sha256"]:
        raise ValueError("Frozen evaluation contract differs")
    dense = read_json(inputs / "dense/manifest.json")
    for name in ("manifest.json", "index.faiss", "vectors.npy", "products.parquet"):
        expected_sha = frozen["return_evidence"]["published_file_hashes"]["bundle/" + name]
        if pins["dense/" + name]["sha256"] != expected_sha:
            raise ValueError("Original M4 bundle differs")
    for name, sha in dense["recipe"]["model_files_sha256"].items():
        if pins["model/" + name]["sha256"] != sha:
            raise ValueError("Original model/tokenizer differs")
    if (
        pins["baseline/per-query.jsonl"]["sha256"]
        != frozen["return_evidence"]["published_file_hashes"]["development/per-query.jsonl"]
    ):
        raise ValueError("Frozen per-query baseline differs")
    validate_baseline(
        pins,
        frozen["return_evidence"]["published_file_hashes"],
        read_json(inputs / "baseline/COMPLETE.json"),
        read_json(inputs / "baseline/report.json"),
    )
    if read_json(inputs / "lexical/manifest.json")["bundle_id"] != expected["m3_bundle_id"]:
        raise ValueError("Frozen lexical reference differs")
    for name in ("train.parquet", "development.parquet", "query_groups.parquet"):
        if pins["prepared/" + name]["sha256"] != audit["output_sha256"][name]:
            raise ValueError("Original supervised split input differs")
    return pins


def real_context(inputs: Path, source: Path, guard: Guard) -> tuple[FeatureContext, dict[str, Any]]:
    manifest = read_json(inputs / "dense/manifest.json")
    dense, products, _ = load_bundle(
        inputs / "dense", manifest["recipe"], expected_origin=manifest["origin"]
    )
    guard.event("dense_native_reconstruction_and_norms_passed", rows=manifest["rows"])
    lexical = load_lexical(inputs / "lexical")
    catalog = pl.read_parquet(
        inputs / "prepared/catalog.parquet",
        columns=["product_locale", "product_id", "product_title"],
    ).sort("product_locale", "product_id")
    if (
        not catalog.equals(products)
        or dense.keys != tuple(lexical.keys)
        or products["product_title"].to_list() != lexical.titles
    ):
        raise ValueError("Canonical catalog/lexical/dense identities or titles differ")
    del catalog, products
    gc.collect()
    spec = configuration(source / "configs/dense.json")
    spec["model_path"] = str(
        inputs / "model"
    )  # Path relocation only; model/recipe bytes stay pinned.
    encoder = PinnedEncoder(spec)
    for field in (
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
        "encoder_code_sha256",
    ):
        if encoder.recipe[field] != manifest["recipe"][field]:
            raise ValueError(f"Frozen product/query scoring recipe differs: {field}")
    binding = DenseBinding(
        manifest["artifact_id"],
        manifest["origin"]["catalog_sha256"],
        digest(list(zip(lexical.keys, lexical.titles, strict=True))),
        manifest["recipe_id"],
        model_identity(encoder.recipe),
    )
    context = FeatureContext(lexical, dense, encoder, binding)
    guard.event("real_encoder_and_catalog_binding_passed")
    return context, manifest


def query_frames(frame: pl.DataFrame) -> list[tuple[int, pl.DataFrame]]:
    return [
        (int(key[0]), rows)
        for key, rows in frame.sort("query_id", "product_locale", "product_id")
        .partition_by("query_id", as_dict=True, maintain_order=True)
        .items()
    ]


def product_keys(frame: pl.DataFrame) -> list[tuple[str, str]]:
    return [
        (str(loc), str(pid))
        for loc, pid in frame.select("product_locale", "product_id").iter_rows()
    ]


def boundary_parity(
    context: FeatureContext,
    ranker: DeterministicRanker,
    cases: list[tuple[int, str, list[tuple[str, str]]]],
    lexical_path: Path,
    destination: Path,
    guard: Guard,
) -> dict[str, Any]:
    model = ranker.model
    if not isinstance(model, ZeroUpdateModel):
        raise ValueError("Actual native prediction adapter required")
    if isinstance(model, FittedPredictor):
        model.record = True
    observations: list[dict[str, Any]] = []
    with TestClient(create_app(lexical_path, ranking=RankingRuntime(context, ranker))) as client:
        if client.get("/health/ready").json().get("ready") is not True:
            raise ValueError("Real HTTP ranking runtime failed readiness")
        for qid, query, keys in cases:
            model.observations.clear()
            offline = offline_predict(context, ranker, query, keys)
            response = client.post(
                "/rerank",
                json={
                    "query": query,
                    "product_ids": [k[1] for k in reversed(keys)],
                    "top_k": min(100, len(keys)),
                },
            )
            if response.status_code != 200 or len(model.observations) != 2:
                raise ValueError(
                    f"Real HTTP prediction failed: {response.status_code}: {response.text}"
                )
            a, b = model.observations
            assert_exact_parity(a, b)
            actual = [
                ((h["product_locale"], h["product_id"]), h["score"])
                for h in response.json()["results"]
            ]
            if actual != offline:
                raise ValueError("Exact offline/HTTP prediction ordering or values differ")
            observations.append(
                {
                    "query_id": qid,
                    "query": query,
                    "keys": a.keys,
                    "names": a.names,
                    "dtypes": a.feature_types,
                    "values": a.values.tolist(),
                    "missing": a.missing.tolist(),
                    "text_empty": a.text_empty.tolist(),
                    "query_fingerprint": a.query_fingerprint,
                    "provenance": asdict(a.provenance),
                    "schema_id": a.schema_id,
                    "offline_http_exact": True,
                    "predictions": actual,
                }
            )
            guard.check()
        # The same API's live search remains bound to the original lexical scores.
        query = cases[0][1]
        search = client.post("/search", json={"query": query, "top_k": 10})
        if search.status_code != 200:
            raise ValueError("Existing BM25 HTTP search unavailable")
        expected_hits = context.lexical.search(query, 10)
        actual_hits = search.json()["results"]
        if [(h["product_locale"], h["product_id"], h["score"]) for h in actual_hits] != [
            (h.product_locale, h.product_id, h.score) for h in expected_hits
        ]:
            raise ValueError("Existing BM25 HTTP search differs from frozen scorer")
    if isinstance(model, FittedPredictor):
        model.record = False
    model.observations.clear()
    result = {
        "status": "passed",
        "cases": observations,
        "native_post_dmatrix": True,
        "atol": 0,
        "rtol": 0,
    }
    write_once(destination, result)
    return result


def deterministic_cases(
    context: FeatureContext, development: pl.DataFrame, guard: Guard
) -> tuple[list[tuple[int, str, list[tuple[str, str]]]], dict[str, Any]]:
    cases: list[tuple[int, str, list[tuple[str, str]]]] = []
    short: dict[str, Any] | None = None
    outside = False
    dense_seconds = []
    evaluation_seconds = []
    for qid, frame in query_frames(development)[:256]:
        query = str(frame["query"][0])
        scores = context.lexical.score_all(query)
        hits = context.lexical.hits_from_scores(scores, 100)
        if len(hits) < 100 and short is None:
            short = {"query_id": qid, "bm25_returned": len(hits)}
        if len(cases) < 4 or (short is not None and short["query_id"] == qid):
            evaluation_started = time.monotonic()
            vector = context.encoder.encode([query])
            started = time.monotonic()
            dense = context.dense.search(vector, 100)  # type: ignore[attr-defined]
            dense_seconds.append(time.monotonic() - started)
            retrieved = {(h.product_locale, h.product_id) for h in hits} | {key for key, _ in dense}
            zeros = np.flatnonzero(scores == 0)
            absent = next((key for key in context.lexical.keys if key not in retrieved), None)
            if not len(zeros) or absent is None:
                raise ValueError("Cannot establish real zero/absent-retrieval edge cases")
            keys = sorted(
                set(product_keys(frame)[:6] + [context.lexical.keys[int(zeros[0])], absent])
            )
            outside = outside or absent not in retrieved
            cases.append((qid, query, keys))
            lexical_keys = [(h.product_locale, h.product_id) for h in hits]
            pool = [
                key
                for key, _ in reciprocal_rank_fusion([lexical_keys, [key for key, _ in dense]], 100)
            ]
            build_features(context, query, product_keys(frame))
            build_features(context, query, pool)
            evaluation_seconds.append(time.monotonic() - evaluation_started)
        if len(cases) >= 4 and short is not None:
            break
        guard.check()
    if short is None or not outside:
        raise ValueError("Required real short-retrieval/absent-top100 coverage unavailable")
    return cases, {
        "short_retrieval": short,
        "products_absent_independent_top100": outside,
        "legitimate_zero_bm25": True,
        "measured_dense_search_seconds": dense_seconds,
        "measured_a_plus_c_query_seconds": evaluation_seconds,
    }


def objective_sanity(params: dict[str, Any]) -> dict[str, Any]:
    import xgboost as xgb

    # Synthetic update sanity only; no development judgments or complete real-data fit.
    x = np.tile(
        np.array(
            [[0, 0, 0, 4, 4], [1, 0.2, 0.1, 4, 4], [2, 0.5, 0.5, 4, 4], [3, 0.9, 1, 4, 4]],
            dtype=np.float32,
        ),
        (128, 1),
    )
    labels = np.tile(np.array([0, 1, 10, 100], dtype=np.float32), 128)
    data = xgb.DMatrix(
        x, label=labels, feature_names=list(FEATURE_NAMES), feature_types=["float"] * 5, nthread=1
    )
    data.set_group([4] * 128)
    booster = xgb.Booster(params=params, cache=[data])
    before = booster.predict(data)
    for iteration in range(2):
        booster.update(data, iteration)
    after = booster.predict(data)
    trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")]
    if (
        not any("split" in tree for tree in trees)
        or np.array_equal(before, after)
        or np.ptp(after) == 0
    ):
        raise ValueError("Objective update sanity failed: no genuine learned tree splits")
    return {
        "synthetic_only": True,
        "updates": 2,
        "genuine_split": True,
        "changed_predictions": True,
        "predictions_first_group": after[:4].tolist(),
        "trees": trees,
    }


def fit_once(
    context: FeatureContext,
    train: pl.DataFrame,
    config: dict[str, Any],
    inputs: dict[str, Any],
    output: Path,
    guard: Guard,
    evaluation_reserve_seconds: float,
    execution_provenance: dict[str, Any],
) -> XGBoostRanker:
    import xgboost as xgb

    write_once(output / "objective-sanity.json", objective_sanity(config["params"]))
    frames = query_frames(train)
    matrix = np.empty((train.height, 5), dtype=np.float32)
    labels = np.empty(train.height, dtype=np.float32)
    group_sizes = []
    offset = 0
    started = time.monotonic()
    for number, (_, frame) in enumerate(frames, 1):
        keys = product_keys(frame)
        batch = build_features(context, str(frame["query"][0]), keys)
        judgments = dict(zip(keys, frame["esci_label"].to_list(), strict=True))
        stop = offset + len(batch.keys)
        matrix[offset:stop] = batch.values
        labels[offset:stop] = [GRADES[judgments[key]] for key in batch.keys]
        group_sizes.append(len(batch.keys))
        offset = stop
        if number == 32:
            estimated = (time.monotonic() - started) / 32 * len(frames)
            remaining = guard.wall_seconds - (time.monotonic() - guard.started)
            if estimated + evaluation_reserve_seconds + 60 > remaining:
                raise RuntimeError(
                    "Measured training feature workload leaves insufficient evaluation time"
                )
        if number % 100 == 0:
            guard.event("training_features", groups=number, total_groups=len(frames))
        guard.check()
    if offset != train.height or set(labels) != {0, 1, 10, 100}:
        raise ValueError("Incomplete supervised feature/grade matrix")
    dtrain = xgb.DMatrix(
        matrix,
        label=labels,
        feature_names=list(FEATURE_NAMES),
        feature_types=["float"] * 5,
        missing=float("nan"),
        nthread=1,
    )
    dtrain.set_group(group_sizes)
    if int(dtrain.get_uint_info("group_ptr")[-1]) != train.height or not np.array_equal(
        dtrain.get_data().toarray(), matrix
    ):
        raise ValueError("Training DMatrix rows/groups/values differ")
    write_once(
        output / "training-features.json",
        {
            "rows": train.height,
            "groups": len(group_sizes),
            "features": list(FEATURE_NAMES),
            "matrix_sha256": digest(matrix.tobytes().hex()),
            "labels_sha256": digest(labels.tobytes().hex()),
            "group_sizes": group_sizes,
            "label_counts": dict(Counter(str(int(x)) for x in labels)),
            "feature_seconds": time.monotonic() - started,
            "dtype": "float32",
            "missing_count": 0,
        },
    )
    guard.event("all_training_features_ready", rows=offset)
    remaining = guard.wall_seconds - (time.monotonic() - guard.started)
    require_fit_budget(remaining, evaluation_reserve_seconds)
    if psutil.virtual_memory().available < 1024**3 + matrix.nbytes * 20:
        raise RuntimeError("Insufficient measured native training memory headroom before fit")
    write_once(
        output / "fit-admission.json",
        {
            "remaining_seconds": remaining,
            "evaluation_reserve_seconds": evaluation_reserve_seconds,
            "available_ram_bytes": psutil.virtual_memory().available,
            "matrix_bytes": matrix.nbytes,
        },
    )
    # This immutable marker is written before the only complete real-data fit call.
    write_once(
        output / "FIT_STARTED.json",
        {
            "configuration": config,
            "rows": train.height,
            "groups": len(group_sizes),
            "retry_allowed": False,
        },
    )
    fit_started = time.monotonic()
    booster = xgb.train(config["params"], dtrain, num_boost_round=100)
    fit_seconds = time.monotonic() - fit_started
    trees = booster.get_dump(dump_format="json")
    train_prediction = booster.predict(dtrain)
    if (
        booster.num_boosted_rounds() != 100
        or not any('"split"' in tree for tree in trees)
        or np.ptp(train_prediction) == 0
    ):
        raise ValueError("Fixed full fit performed no genuine tree updates")
    model_dir = output / "model"
    model_dir.mkdir(exist_ok=False)
    booster.save_model(model_dir / "model.ubj")
    metadata = seal(
        model_dir,
        {
            "schema_version": 1,
            "kind": "atlasrank.xgboost_ranker",
            "configuration": config,
            "provenance": asdict(context.provenance),
            "feature_schema": SCHEMA,
            "feature_schema_id": SCHEMA_ID,
            "xgboost_version": xgb.__version__,
            "numpy_version": np.__version__,
            "input_pins": inputs,
            "execution_provenance": execution_provenance,
            "fit_seconds": fit_seconds,
            "rounds": 100,
            "genuine_tree_updates": True,
        },
        {"model.ubj"},
    )
    result = XGBoostRanker.load(model_dir, context.provenance)
    if not isinstance(result.model, FittedPredictor):
        raise ValueError("Learned native adapter unavailable after save/load")
    if result.model.booster is None or not np.array_equal(
        train_prediction, result.model.booster.predict(dtrain)
    ):
        raise ValueError("Actual fitted-model save/load predictions differ")
    guard.event(
        "single_fit_saved_and_loaded", fit_seconds=fit_seconds, model_id=metadata["artifact_id"]
    )
    return result


def evaluate(
    context: FeatureContext,
    ranker: XGBoostRanker,
    development: pl.DataFrame,
    inputs: Path,
    output: Path,
    guard: Guard,
) -> dict[str, Any]:
    baseline: dict[int, dict[str, Any]] = {}
    for line in (inputs / "baseline/per-query.jsonl").open():
        row = json.loads(line)
        baseline[int(row["query_id"])] = {
            "metrics": row["metrics"]["A"],
            "group": row["group_sha256"],
            "bm25_scores": row["A_scores"]["bm25"],
        }
    results: dict[str, dict[int, Any]] = {"A": {}, "C_reference": {}, "C_ranker": {}}
    groups: dict[int, str] = {}
    unknown = 0
    total_candidates = 0
    with (output / "per-query.jsonl").open("x") as saved:
        for number, (qid, frame) in enumerate(query_frames(development), 1):
            query = str(frame["query"][0])
            groups[qid] = str(frame["group_sha256"][0])
            keys = product_keys(frame)
            judgments = dict(zip(keys, frame["esci_label"].to_list(), strict=True))
            if qid not in baseline or baseline[qid]["group"] != groups[qid]:
                raise ValueError("Frozen development query/group baseline differs")
            a = offline_predict(context, ranker, query, keys)
            results["A"][qid] = score_labeled_candidates(judgments, a, context.lexical.catalog)
            scores = context.lexical.score_all(query)
            actual_bm25 = [
                [list(key), score] for key, score in context.lexical.candidate_scores(scores, keys)
            ]
            if actual_bm25 != baseline[qid]["bm25_scores"]:
                raise ValueError("Frozen M3 judged-candidate BM25 scores changed")
            lexical = [
                (h.product_locale, h.product_id)
                for h in context.lexical.hits_from_scores(scores, 100)
            ]
            vector = context.encoder.encode([query])
            dense = [key for key, _ in context.dense.search(vector, 100)]  # type: ignore[attr-defined]
            pool = [key for key, _ in reciprocal_rank_fusion([lexical, dense], 100)]
            reranked = offline_predict(context, ranker, query, pool)
            after = [key for key, _ in reranked]
            assert_candidate_membership(pool, after)
            before_metrics = score_query(judgments, pool, context.lexical.catalog)
            after_metrics = score_query(judgments, after, context.lexical.catalog)
            if before_metrics.known_exact_recall_at_100 != after_metrics.known_exact_recall_at_100:
                raise ValueError("Fixed-pool Recall@100 changed")
            results["C_reference"][qid], results["C_ranker"][qid] = before_metrics, after_metrics
            unknown += sum(key not in judgments for key in pool)
            total_candidates += len(pool)
            saved.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "group_sha256": groups[qid],
                        "A_ranking": a,
                        "C_before": pool,
                        "C_after": reranked,
                        "candidate_membership_exact": True,
                        "metrics": {name: asdict(values[qid]) for name, values in results.items()},
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            if number % 50 == 0:
                guard.event(
                    "development_evaluation",
                    queries=number,
                    total_queries=development["query_id"].n_unique(),
                )
            guard.check()
    if set(baseline) != set(groups):
        raise ValueError("Frozen development query membership differs")
    intervals = {}
    for field in ("ndcg_at_10", "mrr_at_10"):
        variant = {q: getattr(r, field) for q, r in results["A"].items()}
        for method in ("bm25", "dense", "hybrid"):
            reference = {q: row["metrics"][method][field] for q, row in baseline.items()}
            intervals[f"A.{field}.ranker_minus_frozen_{method}"] = asdict(
                paired_interval(reference, variant, groups)
            )
            guard.check()
    for field in (
        "ndcg_at_10",
        "mrr_at_10",
        "known_exact_recall_at_20",
        "known_exact_recall_at_100",
    ):
        reference = {q: getattr(r, field) for q, r in results["C_reference"].items()}
        variant = {q: getattr(r, field) for q, r in results["C_ranker"].items()}
        intervals[f"C.{field}.ranker_minus_same_pool_rrf"] = asdict(
            paired_interval(reference, variant, groups)
        )
        guard.check()
    return {
        "A": macro_average(list(results["A"].values())),
        "frozen_A": read_json(inputs / "baseline/report.json")["A"],
        "B": {
            "status": "frozen_reference_unchanged",
            "metrics": read_json(inputs / "baseline/report.json")["B"],
        },
        "C": {
            "same_pool_rrf": macro_average(list(results["C_reference"].values())),
            "ranker": macro_average(list(results["C_ranker"].values())),
            "candidate_membership_exact_all_queries": True,
            "recall_at_100_unchanged_all_queries": True,
            "unknown_candidates": unknown,
            "total_candidates": total_candidates,
        },
        "paired_intervals": intervals,
        "per_query_sha256": file_hash(output / "per-query.jsonl"),
        "unknown_policy": (
            "Unknown products retain ranks; known-exact recall is incomplete catalog evidence"
        ),
    }


def run(inputs: Path, source: Path, output: Path, wall_seconds: int, rss_limit: int) -> int:
    guard = Guard(output, wall_seconds, rss_limit)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "fits_executed": 0,
        "official_test_judgments_accessed": False,
        "model_promoted": False,
    }
    try:
        if sys.platform != "linux":
            raise RuntimeError("Real encoder/native worker requires Linux")
        config = fixed_configuration(read_json(source / "configs/m6-ranker-fixed-v1.json"))
        if (
            config["feature_schema_id"] != SCHEMA_ID
            or config["features"] != list(FEATURE_NAMES)
            or config["label_gains"] != GRADES
            or config["bootstrap"] != {"resamples": 2000, "seed": 42, "confidence": 0.95}
            or config["early_stopping"] is not False
        ):
            raise ValueError("Frozen feature/grade/bootstrap contract differs")
        report["environment"] = {
            "python": sys.version,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        }
        report["source_snapshot"] = source_snapshot(source)
        required_versions = {
            "torch": "2.14.1+cpu",
            "sentence-transformers": "6.1.0",
            "transformers": "5.20.2",
            "tokenizers": "0.23.3",
            "safetensors": "0.8.0",
            "faiss-cpu": "1.15.1",
            "xgboost-cpu": "3.1.1",
            "numpy": "2.5.4",
            "scipy": "1.18.1",
        }
        if any(
            importlib.metadata.version(name) != version
            for name, version in required_versions.items()
        ):
            raise ValueError("Temporary CPU runtime differs from approved exact pins")
        pins = verified_inputs(inputs, source)
        report["input_pins"] = pins
        guard.event("all_original_input_bytes_verified")
        context, dense_manifest = real_context(inputs, source, guard)
        import torch
        import xgboost as xgb

        if torch.version.cuda is not None or torch.get_num_threads() != 1:
            raise ValueError("CPU-only single-thread Torch runtime required")
        libraries = {
            "xgboost": Path(xgb.core._LIB._name),
            "torch_cpu": Path(torch.__file__).parent / "lib/libtorch_cpu.so",
        }
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("faiss.") and path and str(path).endswith(".so"):
                libraries[name] = Path(path)
        report["native_library_sha256"] = {
            name: file_hash(path) for name, path in libraries.items()
        }
        report["product_recipe"], report["query_recipe"] = (
            dense_manifest["recipe"],
            context.encoder.recipe,
        )
        train = load_judgments(inputs / "prepared", "train")
        development = load_judgments(inputs / "prepared", "development")
        report["supervision"] = validate_groups(train, development)
        assignments = pl.read_parquet(inputs / "prepared/query_groups.parquet")
        group_fields = ["product_locale", "query_id", "group_sha256", "partition"]
        for name, frame in (("train", train), ("development", development)):
            declared = assignments.filter(pl.col("partition") == name).select(group_fields).unique()
            actual = frame.select(group_fields).unique()
            if not actual.sort(group_fields).equals(declared.sort(group_fields)):
                raise ValueError("Original query-group assignment membership differs")
        del assignments
        audit = read_json(inputs / "prepared/audit.json")
        for partition in ("train", "development"):
            expected = audit["partitions"][partition]
            actual = report["supervision"][partition]
            if (
                actual["rows"] != expected["rows"]
                or actual["groups"] != expected["groups"]
                or actual["labels"] != expected["label_counts"]
            ):
                raise ValueError("Complete original judged query groups/grades differ")
        cases, coverage = deterministic_cases(context, development, guard)
        report["real_edge_coverage"] = coverage
        fixture = DeterministicRanker(context.provenance, "real-m4-native-zero-update-gate")
        fixture.model = ZeroUpdateModel()
        boundary_parity(
            context, fixture, cases, inputs / "lexical", output / "real-boundary.json", guard
        )
        del fixture
        report["phase_1"] = "passed_actual_real_encoder_post_dmatrix_offline_http"
        # Measured full-catalog query cost must leave time for all development queries.
        estimate = (
            max(coverage["measured_a_plus_c_query_seconds"])
            * development["query_id"].n_unique()
            * 1.5
        ) + 180
        remaining = guard.wall_seconds - (time.monotonic() - guard.started)
        require_fit_budget(remaining, estimate)
        guard.event("all_training_gates_passed", estimated_dense_evaluation_seconds=estimate)
        ranker = fit_once(
            context,
            train,
            config,
            pins,
            output,
            guard,
            estimate,
            {
                "source_snapshot": report["source_snapshot"],
                "environment": report["environment"],
                "native_library_sha256": report["native_library_sha256"],
            },
        )
        report["fits_executed"] = 1
        del train
        gc.collect()
        boundary_parity(
            context, ranker, cases, inputs / "lexical", output / "fitted-boundary.json", guard
        )
        report["evaluation"] = evaluate(context, ranker, development, inputs, output, guard)
        report["model"] = read_json(output / "model/manifest.json")
        report["status"] = "completed_single_fixed_fit_and_development_evaluation"
        guard.event("completed")
    except Exception as error:
        report["status"] = "stopped_with_preserved_diagnostic"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        report["fits_executed"] = int((output / "FIT_STARTED.json").exists())
        print(traceback.format_exc(), flush=True)
    finally:
        report["wall_seconds"] = time.monotonic() - guard.started
        report["peak_rss_bytes"] = peak_rss_bytes()
        report["evidence_sha256"] = {
            str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()
        }
        write_once(output / "report.json", report)
    return 0 if report["status"].startswith("completed_") else 2
