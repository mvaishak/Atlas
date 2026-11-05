"""Synthetic-only M8 selection and frozen-metric replay contracts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl
import pytest


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return (
        importlib.import_module("prepare_m8_workload"),
        importlib.import_module("analyze_m8_quality"),
    )


def test_selection_deterministic_deduplicated_and_bounded(
    modules: tuple[ModuleType, ModuleType],
) -> None:
    workload, _ = modules
    frame = pl.DataFrame(
        {
            "query_id": [8, 2, 3, 4, 5, 6, 7, 9, 10],
            "query": [
                " Ａlpha  ",
                "alpha",
                "beta",
                "gamma",
                " ",
                "!",
                "x" * 513,
                "a " * 129,
                "beta",
            ],
        }
    )
    selected, counts = workload.select_queries(frame, 3)
    assert workload.select_queries(frame.reverse(), 3) == (selected, counts)
    assert {row["query_id"] for row in selected} == {2, 3, 4}
    assert len({row["query_sha256"] for row in selected}) == 3
    assert counts == {
        "source_unique_query_ids": 9,
        "eligible_normalized_groups": 3,
        "empty": 1,
        "characters_over_512": 1,
        "analyzer_tokens_outside_1_128": 2,
        "duplicate_normalized_group": 2,
    }
    assert workload.select_queries(frame, 2)[0] == selected[:2]
    with pytest.raises(ValueError, match="Insufficient"):
        workload.select_queries(frame, 4)


def test_identity_rejects_conflicting_strings(modules: tuple[ModuleType, ModuleType]) -> None:
    workload, _ = modules
    with pytest.raises(ValueError, match="multiple"):
        workload.select_queries(pl.DataFrame({"query_id": [1, 1], "query": ["a", "b"]}), 1)
    selected, counts = workload.select_queries(
        pl.DataFrame({"query_id": [1, 1], "query": ["a", "a"]}), 1
    )
    assert len(selected) == counts["source_unique_query_ids"] == 1


def test_development_pin_and_fresh_output(
    modules: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    workload, _ = modules
    source = tmp_path / "development.parquet"
    pl.DataFrame({"query_id": [1], "query": ["a"], "forbidden_label": ["unused"]}).write_parquet(
        source
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"output_sha256": {source.name: workload.sha256(source)}}))
    frame, digest = workload.development_queries(source, audit)
    assert frame.columns == ["query_id", "query"]
    assert digest == workload.sha256(source)
    with pytest.raises(ValueError, match="explicit development"):
        workload.development_queries(tmp_path / "test.parquet", audit)
    audit.write_text(json.dumps({"output_sha256": {source.name: "wrong"}}))
    with pytest.raises(ValueError, match="SHA256"):
        workload.development_queries(source, audit)
    output = tmp_path / "output.json"
    assert workload.write_fresh_json(output, {"a": 1}) == workload.sha256(output)
    with pytest.raises(FileExistsError):
        workload.write_fresh_json(output, {"a": 2})


def replay_fixture(quality: ModuleType) -> tuple[list[dict[str, Any]], dict[int, str]]:
    identities = {1: "one", 2: "two tokens", 3: "four tokens in query"}
    rows = [
        {
            "query_id": i,
            "group_sha256": quality.frozen_group_digest(query),
            "candidate_membership_exact": True,
            "metrics": {
                "C_reference": {"ndcg_at_10": 0.5},
                "C_ranker": {"ndcg_at_10": 0.5 + (i - 2) / 10},
            },
        }
        for i, query in identities.items()
    ]
    return rows, identities


def test_quality_replays_saved_metrics(modules: tuple[ModuleType, ModuleType]) -> None:
    _, quality = modules
    rows, identities = replay_fixture(quality)
    summary = quality.summarize_rows(rows, identities, 3)
    assert summary["mean_C_delta_ndcg_at_10"] == pytest.approx(0)
    assert [x["query_id"] for x in summary["worst_10_query_ids"]] == [1, 2, 3]
    assert [x["query_id"] for x in summary["best_10_query_ids"]] == [3, 2, 1]
    assert {k: v["queries"] for k, v in summary["query_length_slices"].items()} == {
        "1": 1,
        "2-3": 1,
        "4+": 1,
    }


@pytest.mark.parametrize("damage", ["id", "group", "membership", "metric", "count"])
def test_quality_fails_closed(modules: tuple[ModuleType, ModuleType], damage: str) -> None:
    _, quality = modules
    rows, identities = replay_fixture(quality)
    if damage == "id":
        rows[1]["query_id"] = 1
    elif damage == "group":
        rows[1]["group_sha256"] = "wrong"
    elif damage == "membership":
        rows[1]["candidate_membership_exact"] = False
    elif damage == "metric":
        rows[1]["metrics"]["C_ranker"]["ndcg_at_10"] = float("nan")
    else:
        rows.pop()
    with pytest.raises(ValueError):
        quality.summarize_rows(rows, identities, 3)


def test_quality_preserves_original_split_group_encoding(
    modules: tuple[ModuleType, ModuleType],
) -> None:
    workload, quality = modules
    assert quality.frozen_group_digest(" ＯＮＥ ") == (
        "e0f0679aebdc1e2e4aac30c81c7c5297e4feea823c81af058fa13296aa32ded5"
    )
    rows, identities = replay_fixture(quality)
    rows[0]["group_sha256"] = workload.query_digest(identities[1])
    with pytest.raises(ValueError, match="group mismatch"):
        quality.summarize_rows(rows, identities, 3)


def test_quality_length_uses_evaluation_whitespace_contract(
    modules: tuple[ModuleType, ModuleType],
) -> None:
    _, quality = modules
    identities = {1: "alpha-beta"}
    rows = [
        {
            "query_id": 1,
            "group_sha256": quality.frozen_group_digest(identities[1]),
            "candidate_membership_exact": True,
            "metrics": {"C_reference": {"ndcg_at_10": 0.4}, "C_ranker": {"ndcg_at_10": 0.5}},
        }
    ]
    # The retrieval analyzer splits this into two tokens; EVALUATION.md counts one.
    summary = quality.summarize_rows(rows, identities, 1)
    assert summary["query_length_slices"]["1"]["queries"] == 1
    assert summary["query_length_slices"]["2-3"]["queries"] == 0
    assert summary["worst_10_query_ids"][0]["query_length_slice"] == "1"


def test_quality_checks_frozen_file_hash_before_replay(
    modules: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    workload, quality = modules
    development = tmp_path / "development.parquet"
    pl.DataFrame({"query_id": [1], "query": ["one"]}).write_parquet(development)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"output_sha256": {"development.parquet": workload.sha256(development)}})
    )
    m4 = tmp_path / "m4.json"
    m4.write_text(
        json.dumps(
            {
                "partition": "development",
                "provenance": {
                    "source": {
                        "development_sha256": workload.sha256(development),
                        "prepared_audit_sha256": workload.sha256(audit),
                    }
                },
            }
        )
    )
    m6 = tmp_path / "m6.json"
    m6.write_text(
        json.dumps(
            {
                "test_partition": "locked_not_evaluated",
                "development_queries": 4129,
                "evaluation": {"per_query_sha256": "wrong"},
            }
        )
    )
    per_query = tmp_path / "per-query.jsonl"
    per_query.write_text("not valid JSON: must not parse before hash validation")
    with pytest.raises(ValueError, match="per-query source pin mismatch"):
        quality.build_summary(development, audit, m4, m6, per_query)
