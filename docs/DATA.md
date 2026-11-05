# M2 local data contract

M2 is authorized and stops before retrieval or fitting. The repository remains local. [EVALUATION.md](EVALUATION.md) v0.3 is unchanged; its M1-only banner describes the earlier authorization, superseded by the lead's explicit M2 instruction.

## Provenance and acquisition

The source is the [official Amazon ESCI snapshot](https://github.com/amazon-science/esci-data/tree/7916cdf6ab75a462e77f20ab40428a10923998d5), revision `7916cdf6ab75a462e77f20ab40428a10923998d5`. [configs/esci.json](../configs/esci.json) pins sizes and SHA-256 hashes. Parquet hashes/sizes match the revision's Git LFS pointers; GitHub's media endpoint supplies their bytes. Small source/provenance files use the raw-content endpoint. No Git remote is configured to obtain these files.

Acquisition fetches the examples and products Parquet files, query-source CSV, and the snapshot's LICENSE, NOTICE, and README. Apache-2.0 license text and the original attribution remain beside the files. Redistribution needs a later terms review; no data or artifacts are published. Raw files live under `data/esci/raw/<revision>/`, are verified before every preparation, and are made read-only after successful download. Existing mismatches fail without replacement; failed downloads remain as uniquely named `.partial` files.

## Interfaces and schemas

Both commands run from the checkout:

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
uv sync --locked --group dev
uv run --locked python -m atlasrank data-acquire
uv run --locked python -m atlasrank data-prepare --output data/esci/reproduction-a
uv run --locked python -m atlasrank data-prepare --output data/esci/reproduction-b
```

Preparation requires a new output directory inside `data/esci`. It never replaces an output directory. A complete output contains:

| File | Contract |
| --- | --- |
| `catalog.parquet` | Every unique US `(product_locale, product_id)` plus the 5 original text fields; null text becomes an empty string. Missing/blank counts stay in the audit. Source batch/row order is deterministic for the pinned input, but row position is never a product identity |
| `train.parquet`, `development.parquet` | Deduplicated official-train rows with source-row references, original query/product IDs/text/label, normalized query, canonical connected-group key/hash, and assigned partition |
| `quarantine.parquet` | Excluded official-train identities and overlap reason, with relevance labels removed |
| `query_groups.parquet` | Label-free official-train query identity/text-to-group membership, including exclusions |
| `test.references.parquet` | Every selected official-test row in original source-row order, including identity/text and connected-group metadata; no relevance-label column |
| `audit.json` | Schema v1, evaluation contract v0.3, raw provenance, aggregate quality counts, membership counts, and hashes of every output Parquet file |
| `COMPLETE.json` | Schema v1 and the audit SHA-256. Written last; partial output directories lack this completion marker |

`atlasrank.data.prepare.load_judgments(prepared, partition)` accepts only `train` or `development`. It checks completion/audit schema versions and hashes before loading. `test`, `quarantine`, and path-like partition names are rejected. There is no test-evaluation command in M2.

Official test labels remain in the unchanged, checksummed raw file. Preparation projects test identities/text only, and filters to official train before projecting labels. The source has one Parquet row group, so underlying libraries may decode shared pages; application code never receives, counts, evaluates, or uses the test label values. This is a procedural boundary, not encryption or an access-control guarantee. Test label validity/conflicts remain unaudited.

## Deterministic splitting and validation

Select only `small_version == 1` and locale `us`. Reject missing required identities/text, unknown official split values, duplicate example IDs, missing product joins, and conflicting duplicate training labels. Collapse identical training query/product judgments, retaining the lowest source-row reference. A duplicate US catalog product key fails pending review. Preserve all official-test rows.

Normalize query text with NFKC, casefolding, trimming, and whitespace collapse. Union training rows connected by either `(locale, query_id)` or `(locale, normalized text)`, including transitive connections. Quarantine an entire connected group if any member overlaps a test query ID or normalized text. For remaining groups, compute SHA-256 of `42|locale|<smallest normalized text>` in UTF-8; `digest_integer * 5 < 2**256` assigns development. This avoids floating-point threshold rounding. Membership is approximately 80/20 by group and is persisted exactly.

Preparation asserts disjoint identities, normalized text, and group keys at the actual manifest boundary. Synthetic fixtures cover transitive overlap even though the acquired snapshot produced no quarantined groups. Similar paraphrases and product variants remain a residual leakage risk; product overlap is allowed by the approved fixed-catalog design.

To compare complete output bytes:

```sh
uv run --locked python - <<'PY'
from pathlib import Path
from atlasrank.data.common import sha256
a = Path("data/esci/reproduction-a")
b = Path("data/esci/reproduction-b")
assert {p.name: sha256(p) for p in a.iterdir()} == {p.name: sha256(p) for p in b.iterdir()}
PY
```

## Metrics and future supervision

`evaluation.metrics` implements linear-gain NDCG@10, E-only MRR@10, known-exact Recall@20/@100, judgment coverage and returned counts, and equal-query macro averages. IDCG uses the full judged pool; unknown returned products retain their positions with zero observed gain. No-E queries are excluded from recall means with explicit denominator counts; zero-IDCG queries remain in NDCG with score 0. Empty results contribute 0; short rankings aren't padded. Duplicate/unknown products and invalid labels fail. Scores sort descending, then by ascending `(locale, product_id)`.

Track A's wrapper requires the complete judged candidate set. A separate identity check enforces fixed-pool membership across reordered lists. `evaluation.uncertainty` computes paired 95% percentile intervals from 2,000 normalized-group resamples, seed 42, using Python's local `random.Random` and linear-interpolated percentiles. Each sampled group retains all of its query IDs, with equal query weights. Slice summaries retain metric and recall denominators. Missing-title slices mean any judged product has a missing/blank title; no-E groups get an explicit overlap category.

The pinned XGBoost 3.1.1 probe evaluates fixed margins in a zero-tree, CPU `rank:ndcg` Booster. It never calls `train`, `fit`, `update`, or prediction. `ndcg_exp_gain=false` preserves E/S/C/I ratios 100/10/1/0. The `ndcg@10-` metric matches the contract's zero-IDCG score; default `ndcg@10` gives 1 for those groups. Default exponential gain rejects label 100 (>31), proving the explicit gain switch is required. Static and Decimal calculations independently verify the synthetic expectations.

`data.supervision` defines per-query training-only admissibility: E positives, explicitly judged I negatives, and no S/C, unknown, other-query, or held-out negatives. Identical duplicate judgments collapse; conflicts fail. Missing E or I marks a group unusable. These truth tables don't implement a sampler, loss, optimizer, or training pipeline. Actual masks, loss contributions, gradients, and resume checks remain M5 work.

## Resource and failure behavior

The existing limits remain 30 GiB local data storage and 8 GiB preprocessing RSS, with 2 GiB free-disk headroom. Acquisition checks advertised/pinned lengths and reserves remaining download bytes. Preparation reserves twice the source Parquet uncompressed sizes conservatively, processes catalog text in 8,192-row batches, and uses 2 Polars threads. The controller checks child RSS every 0.1 seconds and disk usage every second. A limit violation terminates its worker and preserves partial/history files. Sampling can miss short spikes; these guards aren't an OS-enforced memory sandbox.

Execution records under `data/esci/execution/` include commands, elapsed time, sampled child peak RSS, status, and errors. Failed prepared outputs lack `COMPLETE.json`; don't reuse them as valid bundles. No automatic cleanup deletes raw data or prior outputs. Retained repetitions count toward the storage budget, so request review before more work if the budget is exhausted. See the [M2 review](reviews/M2.md) for executed evidence and the current measured footprint.
