"""Complete supervised groups and fixed-run admission must fail closed."""

from typing import Any

import polars as pl
import pytest


def groups(partition: str, query_id: int, text: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": [query_id] * 4,
            "query": [text] * 4,
            "group_sha256": [text] * 4,
            "partition": [partition] * 4,
            "product_locale": ["us"] * 4,
            "product_id": ["a", "b", "c", "d"],
            "esci_label": ["E", "S", "C", "I"],
        }
    )


def test_supervision_preserves_all_four_grades_and_query_boundaries() -> None:
    from atlasrank.ranking.experiment_contract import validate_groups

    result = validate_groups(groups("train", 1, "apple"), groups("development", 2, "pear"))
    assert result["train"] == {"rows": 4, "groups": 1, "labels": {"E": 1, "S": 1, "C": 1, "I": 1}}


@pytest.mark.parametrize("collision", ["id", "normalized_text", "group"])
def test_development_identity_collisions_are_rejected(collision: str) -> None:
    from atlasrank.ranking.experiment_contract import validate_groups

    train = groups("train", 1, "apple")
    dev = groups("development", 2, "pear")
    field, value = {
        "id": ("query_id", 1),
        "normalized_text": ("query", " ＡＰＰＬＥ  "),
        "group": ("group_sha256", "apple"),
    }[collision]
    dev = dev.with_columns(pl.lit(value).alias(field))
    with pytest.raises(ValueError, match="overlap"):
        validate_groups(train, dev)


def test_duplicate_pairs_and_unknown_grades_are_rejected() -> None:
    from atlasrank.ranking.experiment_contract import validate_groups

    train, dev = groups("train", 1, "apple"), groups("development", 2, "pear")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_groups(pl.concat([train, train.head(1)]), dev)
    with pytest.raises(ValueError, match="grade"):
        validate_groups(train.with_columns(pl.lit("U").alias("esci_label")), dev)


def test_fixed_params_cannot_be_tuned() -> None:
    from atlasrank.ranking.experiment_contract import fixed_configuration

    config: dict[str, Any] = fixed_configuration()
    assert config["params"]["ndcg_exp_gain"] is False
    assert config["params"]["nthread"] == 1
    assert config["num_boost_round"] == 100
    config["params"]["eta"] = 0.1
    with pytest.raises(ValueError, match="fixed"):
        fixed_configuration(config)


def test_fit_requires_reserved_evaluation_time_immediately_before_update() -> None:
    from atlasrank.ranking.experiment_contract import require_fit_budget

    with pytest.raises(RuntimeError, match="evaluation reserve"):
        require_fit_budget(150, 100)
    require_fit_budget(161, 100)


def test_aggregate_baselines_must_match_frozen_pins_and_cross_references() -> None:
    from atlasrank.ranking.experiment_contract import validate_baseline

    files = {
        "development/report.json": "report",
        "development/COMPLETE.json": "complete",
        "development/per-query.jsonl": "queries",
    }
    pins = {"baseline/" + name.split("/")[-1]: {"sha256": value} for name, value in files.items()}
    validate_baseline(pins, files, {"report_sha256": "report"}, {"per_query_sha256": "queries"})
    pins["baseline/report.json"]["sha256"] = "substituted-aggregate"
    with pytest.raises(ValueError, match="Frozen baseline"):
        validate_baseline(
            pins, files, {"report_sha256": "substituted-aggregate"}, {"per_query_sha256": "queries"}
        )
    pins["baseline/report.json"]["sha256"] = "report"
    with pytest.raises(ValueError, match="cross-reference"):
        validate_baseline(
            pins, files, {"report_sha256": "different"}, {"per_query_sha256": "queries"}
        )
