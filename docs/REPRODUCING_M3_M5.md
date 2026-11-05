# Reproducing the M3–M5 experiments

M5.5 consolidates existing experiments. It does not rerun training, catalog encoding or
full evaluation. Start with [the closeout review](reviews/M5.5.md),
[configuration and hashes](../configs/m5-reproduction.json), and
the evidence inventory (`results/m5_5/evidence-inventory.final.json`, private evidence).
EVALUATION.md v0.3 remains byte-for-byte frozen. This guide documents implementation
and provenance; it does not amend the evaluation contract.

## Evidence levels and results

M3 and M4 retain their accepted reports and artifacts. The local M4 per-query report
contains rankings, scores and metrics. M5 training has independently verified returned
weights and checkpoints. M5 Track A has a local aggregate and source script, but no
historical per-query candidate scores/ranks. The manual Colab notebook contains executed
index and Track B summaries. Local M5 compact per-query recalls reproduce the aggregate
and bootstrap calculations; full dense/hybrid rankings and chunk/index bytes remain
external. Aggregate consistency does not independently prove correct retrieval or labels.

| Pipeline | A NDCG@10 | A MRR@10 | B known-exact Recall@20 | B known-exact Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| M3 BM25 | 0.6882263842511278 | 0.7479872331591876 | 0.28326048224440026 | 0.4332656987969319 |
| M4 pretrained dense | 0.7042013548812376 | 0.759239909736398 | 0.25867281972790496 | 0.4331868779517717 |
| M4 pretrained hybrid | 0.7045030184079534 | 0.7649592314523291 | 0.30486595193661387 | 0.48893165355138696 |
| M5 trained dense | 0.7183348379701574 | 0.7818716434664607 | 0.26216427198139103 | 0.430889276151662 |
| M5 trained hybrid | unavailable | unavailable | 0.309062098376579 | 0.49398627992225863 |

All reported evaluations use 4,129 development queries; the local development file has
82,959 judged pairs and 4,129 distinct normalized-query groups. No Track C or official
test results exist. The manual A aggregate also contains judged-pool recall diagnostics:
those are **not** full-catalog Track B recall. The consolidated A summary emits NDCG/MRR
only, plus query and zero-IDCG counts.

| M5 minus M4 | Delta | Paired 95% interval |
| --- | ---: | --- |
| Dense Recall@20 | +0.0034914522534860754 | [-0.0020653572690516784, +0.009122560982251037] |
| Dense Recall@100 | -0.002297601800109688 | [-0.008142866952419964, +0.0035593530718280716] |
| Hybrid Recall@20 | +0.004196146439965167 | [-0.00015640096914199399, +0.008523490185853763] |
| Hybrid Recall@100 | +0.005054626370871675 | [+0.0011339567463601959, +0.008991278255044432] |

The small supported signal is hybrid development Recall@100. Dense standalone
improvement is not statistically established. Hybrid Recall@100 improves on 641 queries,
worsens on 578 and is unchanged on 2,910. Intervals represent query-sampling uncertainty
for one trained run, not training-seed uncertainty, deployment performance or complete
catalog relevance. Multiple comparisons and incomplete judgments further limit conclusions.
No model is promoted by this closeout.

## Exact historical execution differences

The trained model starts from `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Training supervision, exclusions, sampling,
loss, warmup and checkpoint details remain authoritative in `training/`, the original
manifests and [the training review](reviews/M5.training.md). It completed 574 optimizer
updates over 9,180 groups. The loss is logsumexp over all admissible positive/negative
logits minus logsumexp over **all positive** logits, averaged equally over query groups.
Microbatches 8+8 (last 8+4) are weighted by group count; there are no in-batch negatives.

The manual encoding notebook uses Sentence Transformers 6.1.0, float32 weights/outputs,
eager attention, max sequence length 256, right truncation/padding and shared
`normalize_query` semantics (NFKC, casefold, collapsed whitespace). Its actual product
path is `model.encode(..., batch_size=32, normalize_embeddings=True)` over each
1,024-title chunk. ST sorts that whole dispatch by length, uses dynamic padding within
each internal batch, restores input order and normalizes again after the model's
Normalize module. Query/Track A paths explicitly tokenize to fixed 256 and use the
model output directly. The notebook's saved tokenizer JSON does not override ST's
explicit dynamic padding argument. Historical CUDA TF32/determinism policy and the
complete encoding runtime dependency inventory were not recorded; do not invent them.
Future commands disable autocast and retain float32, but cannot establish bitwise CUDA
reproduction of an incompletely recorded historical environment.

M4's product recipe and adapter remain unchanged. Historical M5 is not a strict
weights-only ablation of M4: product padding, batching/extra normalization and dense
retrieval dispatch differ. M5 Track A uses NumPy float32 matrix-vector scoring and
lexicographic product tie-breaking through existing metrics. M4 candidate scoring uses
native FAISS. M5 Track B used FAISS 1.15.1, IndexFlatIP, four OpenMP threads, 16 queries
per search and the returned top-100 order. It did not use M4's cutoff tie refinement.
Do not silently sort those historical dense lists or relabel a new M4-adapter run as
the old M5 experiment. RRF itself matches the existing implementation exactly: independent
lists of up to 100, sum `1/(60+rank)`, then descending score and product-key ties.

## Source and artifact locations

| Artifact | Location / status |
| --- | --- |
| Frozen M3 index | `data/esci/indexes/m3-bm25-v2/`, local |
| Frozen M4 per-query report | `data/esci/reports/m4-cpu-eval-v1/returned/development/per-query.jsonl`, local |
| M4 result archive | `data/esci/reports/m4-cpu-eval-v1/colab-results.zip`, local |
| M5 trained model | `.local/colab/m5-training-return-v1/model-bundle/model/`, verified local inventory |
| Canonical encoding input | `.local/colab/atlasrank-full-input-v4/products.parquet`, local |
| M5 label-free encoding ZIP | `.local/colab/m5-catalog-encoding-v1/m5-encoding-input.zip`, local |
| Manual Track A | `.local/evaluate_m5_track_a.py`, `.local/m5-track-a-results.json`, local |
| Manual BM25 export | `.local/export_m5_bm25.py`, `.local/colab/m5-catalog-encoding-v1/bm25-top100.jsonl`, local |
| Manual Colab source/output notebook | `.local/colab/AtlasRank_M5_Full_Catalog_Encoding.ipynb`, local, preserved |
| M5 compact per-query recall | `.local/colab/m5-catalog-encoding-v1/m5-per-query-compact.json`, local |
| Full M5 chunks/index/rankings | private Drive `MyDrive/AtlasRank/m5/catalog-encoding-v1/`, not mounted/independently read in M5.5 |

The [shared manual-development chat](https://chatgpt.com/share/6ae4145f-6610-83e8-bd81-7432c37c9807)
was retrieved for source audit. An ignored local snapshot is preserved; source snippets
confirm the notebook's index, query, RRF and bootstrap logic. Chat assertions are not
substitutes for native artifact verification. The versioned thin notebook is
[atlasrank_m5_reproduce.ipynb](../notebooks/atlasrank_m5_reproduce.ipynb); it does not
automatically connect a GPU, install dependencies or rerun experiments.

Important SHA-256 values:

| Component | SHA-256 |
| --- | --- |
| M5 weights | `42bfed57ca7c73e457040154aff8c88d6dc4248938e90c08c19e2617a16a2b23` |
| Canonical product projection | `e099d952df2bb83e97fc1d53a7b02d2de8d936b126d1c636ab1369493fa20abf` |
| M5 encoding ZIP | `592504988e429c5b64ac8f48da986131b19d7ff3d9b4802f68a08ed73e411b28` |
| Historical FAISS index, notebook-reported | `562faa580cf37dd3a2095ebd144ef1aa9a7de086795da16a8df1b2fe2441da20` |
| Historical IDs, notebook-reported | `121bd8225394a69889f261cab31a15fa842e75cbec13b3282c36437697ed10ec` |
| Frozen evaluation contract | `884541b50d6bbe6efe128417ef5c3613d3c5ad5913929a71ef1b8888d521e106` |

The original prepared catalog has hash `70b15aa26c285d76d0ffc10a43f04392f514954c401c20fdeb5337bb429ad309`;
it is a different Parquet container/projection from the canonical encoding input. Do not
compare those file hashes as if they were the same serialization. The configuration
also pins each model/tokenizer/pooling file, development labels, BM25 export and M4/M5
per-query source files. No relevance judgments enter the encoding ZIP.

## Lightweight reproduction now

Run from the repository root with the existing `.venv`; do not change Mac dependencies.
Every output below must be a new path. Do not overwrite earlier reports on repetition.

```sh
.venv/bin/python -m atlasrank.experiments.m5 --help
.venv/bin/python -m atlasrank.experiments.m5 verify-inputs \
  --archive .local/colab/m5-catalog-encoding-v1/m5-encoding-input.zip \
  --model .local/colab/m5-training-return-v1/model-bundle/model \
  --catalog .local/colab/atlasrank-full-input-v4/products.parquet \
  --output .local/m5-input-recheck.json
.venv/bin/python scripts/audit_m5_evidence.py --output .local/m5-evidence-recheck.json
.venv/bin/python -m atlasrank.experiments.m5 paired \
  --inputs results/m5_5/paired-inputs.json --output .local/m5-bootstrap-recheck.json
```

The numeric-only versioned paired inputs contain IDs, group hashes and metrics, with
no query text, product identities or relevance labels. They enable standalone statistical
replay. For independently reconstructing them from the private original reports:

```sh
.venv/bin/python -m atlasrank.experiments.m5 collect-pairs \
  --prepared data/esci/prepared-v4 \
  --m4 data/esci/reports/m4-cpu-eval-v1/returned/development/per-query.jsonl \
  --compact .local/colab/m5-catalog-encoding-v1/m5-per-query-compact.json \
  --output .local/m5-pairs-recheck.json
```

This checks exact source hashes, query IDs and group membership before using
`uncertainty.paired_interval` (2,000 grouped resamples, seed 42, percentile 95% CI).
Query eligibility must match. Improved/unchanged counts use the historical `1e-12`
difference threshold; bootstrap deltas are not rounded. Python versus NumPy summation
can change the displayed macro mean's last bit, not the recorded interval.

## Original M3/M4 and M5 training entry points

M3's original commands are `python -m atlasrank bm25-build` and `bm25-evaluate`,
with `--config configs/bm25.json --root data/esci --output NEW_PATH`. The config
points to `m3-bm25-v1`; v1 and the frozen v2 reference have the same bundle ID
`41e8286b586946eb4ec51ce105d4282ccd098316135c97f3501b6edeb76a6a7d`.
Keep the existing build report required by `jobs.evaluate_job`. Existing bundles
must not be overwritten. The consolidated BM25 export command explicitly pins v2.
See [LEXICAL.md](LEXICAL.md) for analyzer arithmetic, endpoint and original commands.

M4's existing `scripts/run_m4_colab_cpu.py` accepts `run --manifest PATH --output PATH`.
The accepted source/input snapshot is retained in `.local/colab/atlasrank-cpu-input-v2/`
and earlier archives; original CPU execution bindings/manifests remain private under
`.local/colab/`. Use that reviewed source snapshot to reproduce the frozen experiment:
the historical source-hash check intentionally rejects an expanded current repository.
A new approved execution binding and empty output are required for a future run;
do not reuse old accounting or substitute current source hashes into old evidence.
[The accepted CPU report](reviews/M4.cpu-run.md) and [CPU instructions](../scripts/run_m4_colab_cpu.py)
remain authoritative. New consolidation commands do not replace the M4 implementation.

M5 training remains in `training/` and `scripts/run_m5_training.py`. Its exact CLI is:

```text
python scripts/run_m5_training.py --archive ARCHIVE --manifest MANIFEST \
  --checkpoints CHECKPOINT_DIR --model MODEL_DIR --input-root INPUT_ROOT \
  --persistent PRIVATE_STORE --allocation-started-utc ACTUAL_START
# Optional recovery flags: --resume CHECKPOINT --recover-interrupted
```

This is a reproduction reference, not an instruction to allocate a GPU in M5.5.
The final executed v8 archive SHA is
`82e7b062eb50e229a541aa57edd131f1f9318b1661478fe00cfa8c36cb1ec596`;
final optimizer checkpoint SHA is
`2b2d042cc7555d39851431a8eb0c12aadad60f700778f27b36bc04a7fe998126`.
Preserve the v6 preparation and failed attempts as well. The original training report
records the bounded execution, losses, model verification and allocation accounting;
this closeout adds no GPU charges or retraining claim.

## Full artifact reproduction commands — not executed in M5.5

Use a previously smoke-tested Linux environment for native FAISS work. Source and model
files must already be private/local, with ST 6.1.0 and FAISS 1.15.1. The current Mac
Torch/FAISS conflict is unresolved. Each command is a separate process: Torch for
Track A/query encoding, FAISS for index validation/evaluation. This is not evidence
that combined Mac imports are safe. Do not install a replacement runtime in this task.

The following shell variables are operator-selected existing paths; `NEW` must not
exist. Full artifact scans/retrieval are substantial work and require execution scope.
No command downloads a model or dataset.

```sh
MODEL=.local/colab/m5-training-return-v1/model-bundle/model
CATALOG=.local/colab/atlasrank-full-input-v4/products.parquet
PREPARED=data/esci/prepared-v4
LEXICAL=data/esci/indexes/m3-bm25-v2
# On Colab these point into the already-existing private Drive tree:
CHUNKS=/content/drive/MyDrive/AtlasRank/m5/catalog-encoding-v1/full-catalog-v1/chunks
INDEX=/content/drive/MyDrive/AtlasRank/m5/catalog-encoding-v1/faiss-index-v1
BM25=.local/colab/m5-catalog-encoding-v1/bm25-top100.jsonl
NEW=.local/m5-reproduction-new
mkdir "$NEW"

python -m atlasrank.experiments.m5 verify-index \
  --catalog "$CATALOG" --chunks "$CHUNKS" --output "$INDEX"
python -m atlasrank.experiments.m5 build-index \
  --catalog "$CATALOG" --chunks "$CHUNKS" --output "$NEW/index"
python -m atlasrank.experiments.m5 track-a --prepared "$PREPARED" \
  --catalog "$CATALOG" --model "$MODEL" --device cpu --output "$NEW/track-a.json"
python -m atlasrank.experiments.m5 export-bm25 --prepared "$PREPARED" \
  --lexical "$LEXICAL" --output "$NEW/bm25-top100.jsonl"
python -m atlasrank.experiments.m5 encode-queries --prepared "$PREPARED" \
  --model "$MODEL" --device cpu --output "$NEW/queries"
python -m atlasrank.experiments.m5 track-b --prepared "$PREPARED" \
  --catalog "$CATALOG" --chunks "$CHUNKS" --index "$INDEX" \
  --queries "$NEW/queries" --bm25 "$BM25" --output "$NEW/track-b.json"
```

Historical-index validation checks expected catalog/model hashes, component hashes,
all 1,188 contiguous chunk ranges, complete 1,215,854-row coverage, canonical identities
and titles, finite float32 normalized vectors, exact native vector reconstruction and
ID alignment. A rebuild creates a new index; exact historical byte pins apply when
validating the original index. A different serialization environment can change rebuilt
bytes; do not replace historical pins. Evaluate a separately rebuilt historical-format
index with `--index-format rebuilt-historical --index NEW_INDEX`; that mode still verifies
all source/chunk/native row alignment and self-checksums, but records a reconstruction
rather than claiming original index bytes. `historical` remains the strict default. Track B records actual native ranking order, index format and complete query manifest
(runtime/recipe and vector hashes), and uses existing metrics/RRF. It requires the exact saved BM25 export hash. The new
Track A command also saves per-query ranks/metrics, filling the old aggregate-only gap
for a future authorized rerun. It has not recreated historical per-query evidence.

Only when encoding is separately authorized, the consolidated encoder can create a
**new** canonical sealed job. It uses the proven chunk/persistence implementation;
it does not pretend the manual historical directory has that schema:

```sh
python -m atlasrank.experiments.m5 encode-catalog --catalog "$CATALOG" \
  --model "$MODEL" --device cuda:0 --output /content/m5-new-workspace \
  --persistent-store /content/drive/MyDrive/AtlasRank/m5/NEW-sealed-run \
  --wall-seconds 3600
python -m atlasrank.experiments.m5 build-canonical-index --catalog "$CATALOG" \
  --chunks /content/m5-new-workspace/job --output "$NEW/canonical-index"
# For such a new bundle, select --index-format canonical and pass its matching job:
# python -m atlasrank.experiments.m5 track-b ... --index-format canonical
```

New sealed jobs bind full model file hashes and distinct catalog/query recipes.
Resume requires the same source, recipe, origin and wall budget; validated completed
chunks are not encoded again. Persistent receipts survive ephemeral workspace loss.
The canonical route adds stronger seals and checks; it does not certify the old
manual run had those protections. Full encoding and evaluation were not run in M5.5.

## Tests and known environmental limits

Run native retrieval and Torch training fixtures in separate processes:

```sh
.venv/bin/ruff check src tests scripts notebooks/atlasrank_m5_reproduce.ipynb
.venv/bin/ruff format --check src/atlasrank/experiments src/atlasrank/evaluation/m5.py \
  src/atlasrank/retrieval/m5_encoder.py src/atlasrank/retrieval/m5_artifacts.py \
  scripts/audit_m5_evidence.py tests/test_m5_reproduction.py \
  tests/test_m5_native_reproduction.py tests/test_colab_full.py \
  notebooks/atlasrank_m5_reproduce.ipynb
.venv/bin/mypy
.venv/bin/python -m pytest -q tests/test_m5_reproduction.py tests/test_m5_native_reproduction.py \
  tests/test_dense_infrastructure.py tests/test_dense.py tests/test_lexical.py tests/test_metrics.py
.venv/bin/python -m pytest -q tests/test_m5_training_sanity.py tests/test_m5_retry_budget.py \
  tests/test_m5_accumulation.py tests/test_m5_operational.py tests/test_m5_budget.py \
  tests/test_dual_encoder_training.py tests/test_training_export.py
.venv/bin/python -m pytest -q tests/test_colab.py tests/test_colab_cpu_adapter.py \
  tests/test_colab_cpu_run.py tests/test_colab_full.py tests/test_data.py \
  tests/test_foundation.py tests/test_search_api.py tests/test_tracking.py
```

Do not invoke a combined Mac `pytest` process that imports both Torch and FAISS.
The CUDA active-dropout resume test is skipped without a GPU; do not count that as
executed CUDA parity. Broad `ruff check .` and formatting checks expose pre-existing
format/import issues in five historical training notebooks; those execution snapshots
were preserved instead of reformatted. Full Linux CI and MPS are still unrun.
