from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from atlasrank.training.dual_encoder import build_sampling_plan
from atlasrank.training.export import _training_groups


def test_export_group_builder_keeps_only_train_e_and_i_and_masks_global_e(tmp_path: Path) -> None:
    rows = [
        (
            1,
            "Wireless Headphones",
            "wireless headphones",
            "wireless headphones",
            "a" * 64,
            "p1",
            "E",
            "train",
        ),
        (
            1,
            "Wireless Headphones",
            "wireless headphones",
            "wireless headphones",
            "a" * 64,
            "p1",
            "E",
            "train",
        ),
        (
            1,
            "Wireless Headphones",
            "wireless headphones",
            "wireless headphones",
            "a" * 64,
            "p2",
            "I",
            "train",
        ),
        (
            1,
            "Wireless Headphones",
            "wireless headphones",
            "wireless headphones",
            "a" * 64,
            "p3",
            "S",
            "train",
        ),
        (
            1,
            "Wireless Headphones",
            "wireless headphones",
            "wireless headphones",
            "a" * 64,
            "p4",
            "C",
            "train",
        ),
        (2, "Phone Case", "phone case", "phone case", "b" * 64, "p2", "E", "train"),
        (2, "Phone Case", "phone case", "phone case", "b" * 64, "n2", "I", "train"),
    ]
    frame = pl.DataFrame(
        rows,
        schema=[
            "query_id",
            "query",
            "normalized_query",
            "group_key",
            "group_sha256",
            "product_id",
            "esci_label",
            "partition",
        ],
        orient="row",
    ).with_columns(pl.lit("us").alias("product_locale"))
    train_path = tmp_path / "train.parquet"
    frame.write_parquet(train_path)
    groups, counts = _training_groups(train_path)
    assert counts == {"C": 1, "E": 3, "I": 2, "S": 1}
    first = next(group for group in groups if group.group_key == "wireless headphones")
    assert first.positives == ("p1",)
    assert first.negatives == ("p2",)
    plan = build_sampling_plan(groups)
    assert all(group.group_key != first.group_key for group in plan.groups)
    assert plan.usable_group_count == 1
    assert len(plan.sha256) == 64


def test_export_group_builder_rejects_non_train_rows(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "query": ["query"],
            "query_id": [1],
            "product_id": ["p1"],
            "product_locale": ["us"],
            "esci_label": ["E"],
            "normalized_query": ["query"],
            "group_key": ["query"],
            "group_sha256": ["a" * 64],
            "partition": ["development"],
        }
    )
    source = tmp_path / "not-train.parquet"
    frame.write_parquet(source)
    with pytest.raises(ValueError, match="train partition"):
        _training_groups(source)
