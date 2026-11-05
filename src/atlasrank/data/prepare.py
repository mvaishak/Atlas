"""Audit pinned raw files and write deterministic local manifests, with no test labels."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from atlasrank.data.common import check_disk, check_memory, sha256, tree_bytes, write_json
from atlasrank.data.splits import Query, assign_queries, normalize_query
from atlasrank.data.supervision import LABELS

TEXT_FIELDS = [
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]
KEYS = ["product_locale", "product_id"]
IDENTITY_COLUMNS = [
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "small_version",
    "split",
]


def require_schema(schema: pl.Schema, required: dict[str, pl.DataType | type[pl.DataType]]) -> None:
    for column, dtype in required.items():
        if column not in schema or schema[column] != dtype:
            raise ValueError(f"Schema mismatch for {column}: expected {dtype}")


def require_nonempty(frame: pl.DataFrame, columns: list[str]) -> None:
    for column in columns:
        series = frame[column]
        if series.null_count() or (
            series.dtype == pl.String and series.str.strip_chars().eq("").any()
        ):
            raise ValueError(f"Null/empty required field: {column}")


def catalog_snapshot(source: Path, destination: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    parquet = pq.ParquetFile(source)
    schema = pl.scan_parquet(source).collect_schema()
    require_schema(schema, {**dict.fromkeys(KEYS + TEXT_FIELDS, pl.String)})
    counts: Counter[str] = Counter()
    missing_nulls: Counter[str] = Counter()
    missing_blank: Counter[str] = Counter()
    seen: set[str] = set()
    keys: list[pl.DataFrame] = []
    writer: Any = None
    total = 0
    try:
        for batch in parquet.iter_batches(batch_size=8192):
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            total += frame.height
            require_nonempty(frame, KEYS)
            counts.update(frame["product_locale"].to_list())
            frame = frame.filter(pl.col("product_locale") == "us").select(KEYS + TEXT_FIELDS)
            ids = frame["product_id"].to_list()
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("Duplicate US catalog product key; review required")
            seen.update(ids)
            for field in TEXT_FIELDS:
                missing_nulls[field] += frame[field].null_count()
                missing_blank[field] += int(
                    frame[field].fill_null("").str.strip_chars().eq("").sum()
                )
            frame = frame.with_columns([pl.col(field).fill_null("") for field in TEXT_FIELDS])
            keys.append(frame.select(KEYS))
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=8192)
            check_memory()
    finally:
        if writer is not None:
            writer.close()
    if not keys or not seen:
        raise ValueError("Empty US catalog")
    return pl.concat(keys), {
        "raw_rows": total,
        "raw_locale_counts": dict(sorted(counts.items())),
        "us_unique_products": len(seen),
        "us_duplicate_keys": 0,
        "us_null_text_fields": dict(sorted(missing_nulls.items())),
        "us_empty_or_whitespace_text_fields_including_nulls": dict(sorted(missing_blank.items())),
        "schema": {name: str(dtype) for name, dtype in schema.items()},
        "uncompressed_parquet_bytes": sum(
            parquet.metadata.row_group(i).total_byte_size for i in range(parquet.num_row_groups)
        ),
    }


def query_records(frame: pl.DataFrame) -> list[Query]:
    unique = frame.select("product_locale", "query_id", "query").unique()
    return [
        Query(str(locale), int(query_id), str(text))
        for locale, query_id, text in unique.iter_rows()
    ]


def assignments_frame(frame: pl.DataFrame, held_out: pl.DataFrame) -> pl.DataFrame:
    assignments = assign_queries(query_records(frame), query_records(held_out))
    if not assignments:
        raise ValueError("No query groups")
    return (
        pl.DataFrame([asdict(a) for a in assignments]).rename({"locale": "product_locale"}).unique()
    )


def prepare(config: Path, root: Path, output: Path) -> Path:
    """Only official-train rows enter the relevance-label projection.

    Official-test output is a label-free row-reference manifest into unchanged raw
    data. Parquet may decode shared pages internally; no test label is consumed,
    counted, emitted, evaluated, or used for split selection by application code.
    """
    spec = json.loads(config.read_text())
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Prepared output must be inside the budgeted data root")
    if output.exists():
        raise FileExistsError("Output already exists; preserve previous preparation history")
    raw = root / "raw" / spec["revision"]
    for entry in spec["files"]:
        path = raw / entry["name"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Pinned raw file mismatch: {entry['name']}")
    # Two complete derived copies fit below 30 GiB. Reserve conservatively before starting.
    estimate = (
        sum(
            pq.ParquetFile(raw / name).metadata.row_group(i).total_byte_size
            for name in ("examples.parquet", "products.parquet")
            for i in range(pq.ParquetFile(raw / name).num_row_groups)
        )
        * 2
    )
    check_disk(root, estimate)
    output.mkdir(parents=True)
    catalog, product_audit = catalog_snapshot(raw / "products.parquet", output / "catalog.parquet")
    check_disk(root)
    source = pl.scan_parquet(raw / "examples.parquet").with_row_index("source_row")
    require_schema(
        source.collect_schema(),
        {
            "example_id": pl.Int64,
            "query": pl.String,
            "query_id": pl.Int64,
            "product_id": pl.String,
            "product_locale": pl.String,
            "small_version": pl.Int64,
            "split": pl.String,
            "esci_label": pl.String,
        },
    )
    selected = source.filter((pl.col("product_locale") == "us") & (pl.col("small_version") == 1))
    identities = selected.select("source_row", *IDENTITY_COLUMNS).collect(engine="streaming")
    if not identities.height:
        raise ValueError("Empty US small-version judgments")
    require_nonempty(identities, IDENTITY_COLUMNS)
    if set(identities["split"].unique().to_list()) != {"train", "test"}:
        raise ValueError("Expected only official train and test assignments")
    if identities["example_id"].n_unique() != identities.height:
        raise ValueError("Duplicate example_id")
    if identities.join(catalog, on=KEYS, how="anti").height:
        raise ValueError("Missing product joins; review required")
    official_train = identities.filter(pl.col("split") == "train")
    test = identities.filter(pl.col("split") == "test")
    assignments = assignments_frame(official_train, test)
    source_audit: dict[str, Any] = {"present": False}
    if (raw / "sources.csv").exists():
        sources = pl.read_csv(raw / "sources.csv")
        require_schema(sources.schema, {"query_id": pl.Int64, "source": pl.String})
        require_nonempty(sources, ["query_id", "source"])
        if sources["query_id"].n_unique() != sources.height:
            raise ValueError("Duplicate source query_id")
        missing_sources = (
            identities.select("query_id")
            .unique()
            .join(sources.select("query_id"), on="query_id", how="anti")
            .height
        )
        if missing_sources:
            raise ValueError("Missing query provenance joins")
        source_audit = {
            "present": True,
            "rows": sources.height,
            "schema": {name: str(dtype) for name, dtype in sources.schema.items()},
            "missing_query_provenance_joins": missing_sources,
        }
    # Filter before projecting esci_label; never return the test label column.
    labels = (
        selected.filter(pl.col("split") == "train")
        .select("source_row", "esci_label")
        .collect(engine="streaming")
    )
    train = official_train.join(labels, on="source_row", validate="1:1")
    require_nonempty(train, ["esci_label"])
    if not set(train["esci_label"].unique().to_list()).issubset(LABELS):
        raise ValueError("Unknown ESCI label in official training data")
    pair_key = ["product_locale", "query_id", "product_id"]
    conflicts = train.group_by(pair_key).agg(pl.col("esci_label").n_unique().alias("labels"))
    if conflicts.filter(pl.col("labels") > 1).height:
        raise ValueError("Conflicting duplicate training judgments; review required")
    train = train.with_columns(
        pl.col("query")
        .map_elements(normalize_query, return_dtype=pl.String)
        .alias("normalized_query")
    ).join(assignments, on=["product_locale", "query_id", "normalized_query"], validate="m:1")
    duplicate_count = train.height - train.unique(subset=pair_key).height
    train = train.sort("source_row").unique(subset=pair_key, keep="first", maintain_order=True)
    test_assignments = assignments_frame(test, test.head(0)).drop("partition")
    test = test.with_columns(
        pl.col("query")
        .map_elements(normalize_query, return_dtype=pl.String)
        .alias("normalized_query")
    ).join(test_assignments, on=["product_locale", "query_id", "normalized_query"], validate="m:1")
    report: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_contract": "v0.3",
        "revision": spec["revision"],
        "catalog": product_audit,
        "query_provenance": source_audit,
        "examples": {
            "raw_rows": pq.ParquetFile(raw / "examples.parquet").metadata.num_rows,
            "schema": {name: str(dtype) for name, dtype in source.collect_schema().items()},
            "raw_identity_null_counts": source.select(pl.col(IDENTITY_COLUMNS).null_count())
            .collect(engine="streaming")
            .to_dicts()[0],
            "official_train_query_ids": official_train["query_id"].n_unique(),
            "official_train_unique_products": official_train["product_id"].n_unique(),
            "train_query_ids_with_multiple_normalized_texts": assignments.group_by("query_id")
            .agg(pl.col("normalized_query").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height,
            "train_normalized_texts_with_multiple_query_ids": assignments.group_by(
                "normalized_query"
            )
            .agg(pl.col("query_id").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height,
            "us_small_rows": identities.height,
            "official_train_rows_before_deduplication": official_train.height,
            "official_train_label_counts": dict(
                sorted(Counter(labels["esci_label"].to_list()).items())
            ),
            "identical_duplicate_train_pairs_collapsed": duplicate_count,
            "conflicting_train_pairs": 0,
            "missing_product_joins": 0,
            "official_test_rows_preserved": test.height,
            "official_test_query_ids": test["query_id"].n_unique(),
            "official_test_unique_products": test["product_id"].n_unique(),
            "official_test_duplicate_pair_rows_preserved": test.height
            - test.unique(subset=pair_key).height,
            "test_labels_accessed_by_application": False,
            "test_labels_audited": False,
        },
        "partitions": {},
        "normalization": "NFKC, casefold, trim, whitespace collapse; connected ID/text components",
        "split_hash": "SHA256(42|locale|smallest normalized text); bottom fifth development",
        "test_policy": (
            "Raw labels sealed in unchanged raw file; manifest contains row references only"
        ),
        "sources": spec,
    }
    assignments.sort("product_locale", "query_id", "normalized_query").write_parquet(
        output / "query_groups.parquet", compression="zstd"
    )
    for partition in ("train", "development", "quarantine"):
        subset = train.filter(pl.col("partition") == partition).sort(pair_key)
        if partition == "quarantine":
            # Exclusion report has identities/reasons only, no supervision for future loaders.
            subset = subset.drop("esci_label").with_columns(
                pl.lit("connected_group_overlaps_test_identity_or_text").alias("exclusion_reason")
            )
        subset.write_parquet(output / f"{partition}.parquet", compression="zstd")
        report["partitions"][partition] = {
            "rows": subset.height,
            "query_ids": subset["query_id"].n_unique(),
            "groups": subset["group_key"].n_unique(),
            "unique_products": subset["product_id"].n_unique(),
            **(
                {"label_counts": dict(sorted(Counter(subset["esci_label"].to_list()).items()))}
                if partition != "quarantine"
                else {}
            ),
        }
    test.sort("source_row").write_parquet(output / "test.references.parquet", compression="zstd")
    # Independent overlap assertions at the actual output boundary.
    partitioned = train.select("query_id", "normalized_query", "group_key", "partition").unique()
    for field in ("query_id", "normalized_query", "group_key"):
        if (
            partitioned.group_by(field)
            .agg(pl.col("partition").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height
        ):
            raise AssertionError(f"Split leakage in {field}")
    admitted = partitioned.filter(pl.col("partition") != "quarantine")
    if admitted.join(test.select("query_id").unique(), on="query_id", how="inner").height:
        raise AssertionError("Train/development overlaps official test IDs")
    if admitted.join(
        test.select("normalized_query").unique(), on="normalized_query", how="inner"
    ).height:
        raise AssertionError("Train/development overlaps official test text")
    check_memory()
    check_disk(root)
    report["output_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.parquet"))}
    report["output_parquet_bytes"] = tree_bytes(output)
    write_json(output / "audit.json", report)
    write_json(
        output / "COMPLETE.json",
        {"schema_version": 1, "audit_sha256": sha256(output / "audit.json")},
    )
    return output


def load_judgments(prepared: Path, partition: str) -> pl.DataFrame:
    """Public development loader fails closed on locked or excluded partitions."""
    if partition not in {"train", "development"}:
        raise ValueError("Only train/development judgments are accessible before final approval")
    complete = json.loads((prepared / "COMPLETE.json").read_text())
    if complete.get("schema_version") != 1:
        raise ValueError("Unsupported completion schema")
    if sha256(prepared / "audit.json") != complete["audit_sha256"]:
        raise ValueError("Preparation report checksum mismatch")
    audit = json.loads((prepared / "audit.json").read_text())
    if audit.get("schema_version") != 1 or audit.get("evaluation_contract") != "v0.3":
        raise ValueError("Unsupported data/evaluation schema")
    name = f"{partition}.parquet"
    if sha256(prepared / name) != audit["output_sha256"][name]:
        raise ValueError("Judgment manifest checksum mismatch")
    return pl.read_parquet(prepared / name)
