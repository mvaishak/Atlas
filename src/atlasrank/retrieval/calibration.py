"""One bounded, label-free CPU throughput pilot; never encode the full catalog."""

import hashlib
import heapq
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import pyarrow.parquet as pq

from atlasrank.data.common import check_disk, check_memory, peak_rss_bytes, sha256, write_json
from atlasrank.data.splits import normalize_query
from atlasrank.retrieval.encoder import PinnedEncoder, configuration, verify_snapshot


def sample_catalog(catalog: Path, count: int) -> pl.DataFrame:
    def records() -> Any:
        for batch in pq.ParquetFile(catalog).iter_batches(
            batch_size=8192, columns=["product_locale", "product_id", "product_title"]
        ):
            for row in batch.to_pylist():
                key = (row["product_locale"], row["product_id"])
                digest = hashlib.sha256(f"42|{key[0]}|{key[1]}".encode()).hexdigest()
                yield digest, *key, row["product_title"]

    selected = heapq.nsmallest(count, records())
    return pl.DataFrame(
        selected,
        schema=["selection_sha256", "product_locale", "product_id", "product_title"],
        orient="row",
    ).sort("product_locale", "product_id")


def actual_model_checks(encoder: PinnedEncoder) -> dict[str, Any]:
    import torch

    texts = [" ＷＩＲＥＬＥＳＳ\t headphones ", "", "café Straße", "apple " * 300]
    batched = encoder.encode(texts)
    repeated = encoder.encode(texts)
    np.testing.assert_array_equal(batched, repeated)
    singles = np.concatenate([encoder.encode([text]) for text in texts])
    np.testing.assert_allclose(batched, singles, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(
        encoder.encode([texts[0]]), encoder.encode(["wireless headphones"])
    )
    features = encoder.model.tokenizer(
        [normalize_query(t) for t in texts],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = encoder.model[0].auto_model(**features).last_hidden_state
        mask = features["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        manual = torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy()
    np.testing.assert_allclose(batched, manual, rtol=1e-5, atol=1e-6)
    return {
        "repeat_exact": True,
        "normalization_exact": True,
        "single_vs_batch_max_absolute_error": float(np.max(np.abs(batched - singles))),
        "manual_masked_mean_normalization_max_absolute_error": float(
            np.max(np.abs(batched - manual))
        ),
        "rtol": 1e-5,
        "atol": 1e-6,
        "synthetic_inputs": len(texts),
        "empty_and_truncated_text_checked": True,
    }


def calibrate(config: Path, root: Path, output: Path) -> Path:
    spec = configuration(config)
    if output.exists():
        raise FileExistsError("Preserve previous calibration output")
    for name in ("model_path", "prepared_path"):
        if not Path(spec[name]).resolve().is_relative_to(root.resolve()):
            raise ValueError("Pilot inputs must stay inside the budgeted data root")
    check_disk(root, spec["pilot_incremental_disk_bytes"])
    if psutil.virtual_memory().available < spec["pilot_available_ram_bytes"]:
        raise RuntimeError("Pilot requires at least 4 GiB available system RAM")
    prepared = Path(spec["prepared_path"])
    if sha256(prepared / "audit.json") != spec["prepared_audit_sha256"]:
        raise ValueError("Prepared audit differs from approved snapshot")
    if sha256(prepared / "catalog.parquet") != spec["catalog_sha256"]:
        raise ValueError("Catalog differs from approved snapshot")
    complete = json.loads((prepared / "COMPLETE.json").read_text())
    if complete["audit_sha256"] != spec["prepared_audit_sha256"]:
        raise ValueError("Incomplete approved preparation")
    model_hashes = verify_snapshot(spec)
    sample = sample_catalog(prepared / "catalog.parquet", spec["pilot_titles"])
    if (
        sample.height != 4096
        or sample.select("product_locale", "product_id").unique().height != 4096
    ):
        raise ValueError("Pilot sample count/identities differ")
    output.mkdir(parents=True)
    sample.write_parquet(output / "sample.parquet", compression="zstd")
    write_json(
        output / "STARTED.json", {"schema_version": 1, "configuration_sha256": sha256(config)}
    )
    loaded = time.perf_counter()
    encoder = PinnedEncoder(spec)
    load_seconds = time.perf_counter() - loaded
    integration = actual_model_checks(encoder)
    titles = sample["product_title"].to_list()
    lengths = [
        len(encoder.model.tokenizer.encode(normalize_query(t), truncation=False, verbose=False))
        for t in titles
    ]
    matrix = np.lib.format.open_memmap(
        output / "vectors.npy", mode="w+", dtype=np.float32, shape=(4096, 384)
    )
    started = time.perf_counter()
    for offset in range(0, 4096, 32):
        if time.perf_counter() - started >= 600:
            raise RuntimeError("Pilot encoding cap exceeded; partial output preserved")
        matrix[offset : offset + 32] = encoder.encode(titles[offset : offset + 32])
        check_memory()
        if psutil.virtual_memory().available < 1024**3:
            raise RuntimeError("System available RAM fell below 1 GiB; partial pilot preserved")
        check_disk(root)
        if (offset + 32) % 512 == 0:
            matrix.flush()
            print(
                json.dumps({"event": "cpu_calibration_progress", "titles": offset + 32}), flush=True
            )
    matrix.flush()
    elapsed = time.perf_counter() - started
    rate = 4096 / elapsed
    report = {
        "schema_version": 1,
        "kind": "bounded CPU throughput calibration; not full-catalog evaluation",
        "model_revision": spec["revision"],
        "model_file_sha256": model_hashes,
        "configuration_sha256": sha256(config),
        "catalog_sha256": spec["catalog_sha256"],
        "sample_sha256": sha256(output / "sample.parquet"),
        "vectors_sha256": sha256(output / "vectors.npy"),
        "titles": 4096,
        "dimensions": 384,
        "batch_size": 32,
        "max_length": 256,
        "device": "cpu",
        "precision": "float32",
        "threads": 1,
        "load_seconds": load_seconds,
        "encoding_seconds": elapsed,
        "titles_per_second": rate,
        "token_length_including_specials": {
            "p50": float(np.percentile(lengths, 50)),
            "p95": float(np.percentile(lengths, 95)),
            "max": max(lengths),
            "truncated_titles": sum(n > 256 for n in lengths),
            "empty_titles": sum(not normalize_query(t) for t in titles),
        },
        "process_peak_rss_bytes": peak_rss_bytes(),
        "actual_model_integration": integration,
        "full_encoding_hours_arithmetic_extrapolation": 1215854 / rate / 3600,
        "extrapolation_caveat": (
            "Sample and thermal conditions may differ; excludes index building, "
            "full evaluation, setup, failures and restarts"
        ),
        "labels_loaded": False,
        "full_catalog_encoded": False,
        "training_performed": False,
    }
    write_json(output / "report.json", report)
    write_json(
        output / "COMPLETE.json",
        {"schema_version": 1, "report_sha256": sha256(output / "report.json")},
    )
    return output
