# M3 lexical search contract

M3 implements one fixed lexical baseline on the full approved US catalog. M2 commit `3f4e2d9ce3409d05134d971fb6e0d90bdb684446` was approved before this work. [EVALUATION.md](EVALUATION.md) v0.3 is unchanged; its historical authorization banner is superseded by the lead's explicit M3 instruction. M4 and learned modeling remain unauthorized.

## Fields, analyzer, and score

Only `product_title` is indexed. Description, bullet points, brand, and color columns aren't concatenated or weighted. The original title is retained for display. All 1,215,854 catalog products remain, including the 5 titles whose text has no analyzer tokens.

The analyzer `nfkc_casefold_unicode_alnum_v1` uses the existing NFKC/casefold/whitespace normalization, then extracts Unicode letter/digit runs with `[^\W_]+`. Punctuation and underscores separate tokens. There are no stopwords, stemming, accent removal, or truncation. Repeated query terms contribute repeatedly. Indexing, offline search, evaluation, and HTTP search call the same analyzer.

The pinned BM25S 0.2.14 NumPy backend uses the Lucene variant with `k1=1.5`, `b=0.75`, float32 scores, and int32 sparse indices. For each query-token occurrence, its contribution is:

```text
log(1 + (N - df + 0.5) / (df + 0.5))
  * tf / (tf + k1 * (1 - b + b * document_length / average_document_length))
```

The library omits the optional `(k1 + 1)` numerator factor used in some BM25 conventions. That constant changes score magnitude; the selected convention is frozen for later comparisons/features. IDF and average length use the full catalog, including empty-token documents. No query labels or held-out statistics fit the index. The [library implementation](https://github.com/xhluca/bm25s) was inspected and synthetic scores were checked against independent arithmetic.

Documents sort by `(product_locale, product_id)` before indexing. Search selects up to 100 **positive-score** matches; OOV or no-token queries return an empty offline result. It never pads a short match list with zero-score documents or uses relevance labels to filter results. Ties, including ties at the cutoff, resolve by ascending product key. Unknown-but-matching catalog products retain their positions for evaluation. Track A separately scores every judged candidate, including zero scores, with the full-catalog IDF statistics.

## Reproduction and files

Use the project lock and approved M2 `data/esci/prepared-v4` snapshot. [configs/bm25.json](../configs/bm25.json) pins its audit, catalog, and development hashes plus the single baseline configuration. It also records the reference index path and fixed basic-latency workload settings. No configuration sweep occurred.

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv sync --locked --group dev
uv run --locked python -m atlasrank bm25-build --output data/esci/indexes/m3-bm25-v1
uv run --locked python -m atlasrank bm25-build --output data/esci/indexes/m3-bm25-v2
uv run --locked python -m atlasrank bm25-evaluate --output data/esci/reports/m3-bm25-v3
```

Those paths already exist in this workspace. New runs require fresh destinations; old files aren't replaced or removed. Replica builds can use new output names with the same config; evaluation always loads the config's reference index. The controller reuses the 8 GiB sampled preprocessing RSS / 30 GiB local data reservation with 2 GiB free headroom. Indexes and reports live within that monitored data root. These sampled guards can miss brief spikes and aren't OS-enforced allocation limits.

A bundle contains 3 sparse `.npy` arrays, vocabulary/parameter JSON, canonical product/title Parquet, `manifest.json`, and a completion marker. The manifest schema is v1 and includes analyzer/fields, math parameters, numerical-library versions, document count, M2/config lineage, and component SHA-256 hashes. Its content digest is the bundle ID; timing measurements stay outside the bundle, allowing deterministic rebuilds. Completion is written last.

Load validates the completion/manifest identity, exact file inventory/hashes, current pinned dependency versions, schema/parameters, array dimensions/dtypes/bounds/finite values, vocabulary IDs, and canonical unique US product mapping. Arrays use read-only memory mapping with `allow_pickle=False`; no pickle model is loaded. Bundles are immutable by application convention and need restart-based selection; there is no hot reload, signing, or adversarial-upload interface.

Build records under `data/esci/execution/` retain stage counts, elapsed time, probe score hashes, bundle size, command status, and sampled RSS. Both full builds produced byte-identical files. Fresh loading also reproduced complete score-vector hashes for fixed synthetic probe queries. Golden tests additionally check full score arrays, input-row permutations, boundary ties, and concurrent request isolation.

## Offline and HTTP interfaces

`LexicalIndex.search(query, k)` returns stable locale/product IDs, original title, and BM25 score. `candidate_scores` supports Track A on the supplied judged pool. Scores express lexical matching under the frozen formula; they aren't probabilities or judgments.

```sh
uv run --locked python -m atlasrank bm25-search --index data/esci/indexes/m3-bm25-v1 --query "wireless headphones" --top-k 10
uv run --locked python -m atlasrank serve --index data/esci/indexes/m3-bm25-v1 --port 8000
curl -s http://127.0.0.1:8000/health/ready
curl -s -H 'Content-Type: application/json' -d '{"query":"wireless headphones","top_k":10}' http://127.0.0.1:8000/search
```

The server binds loopback only. Search accepts exactly `query`, optional integer `top_k` (default 10, range 1–100), and optional `locale` (only `us`). Queries must contain 1–512 characters and 1–128 analyzer tokens. Strict types reject booleans/string top-k values; unknown fields, unsupported locales, blank/punctuation-only queries, and invalid JSON return 422 with `invalid_request`. Valid OOV requests return 200 and no results. All 4,129 development queries fit the HTTP query bounds.

Results include pipeline/bundle IDs, product identities/titles/scores, returned count, and measured search duration. `/health/live` remains 200; `/health/ready` and valid `/search` return 503 with `index_unavailable` if no compatible bundle loads. Expected internal search failures return 500 with `search_failed`, without query/exception details. Startup checks the proposed 6 GiB process ceiling; it doesn't claim continuous OS memory isolation.

Application request logs are JSON with generated request ID, bounded route name/status, bundle ID, and duration. They exclude query text, product IDs, and arbitrary user input. Uvicorn access logging is disabled. The actual-bundle loopback smoke test proved exact HTTP/offline results and verified query text was absent from the server log. The smoke server was shut down afterward. This is a minimal local API, with operational hardening/load testing deferred to later milestones.

## Evaluation and measurement boundaries

Only `development.parquet` is loaded through M2's guarded loader; its hash must match the frozen config. The jobs never open raw examples or official-test references/judgments. A scores all labeled candidates; B retrieves from the full catalog and reports known-exact Recall@20/@100, judgment coverage, and returned counts. Unknown labels remain unknown. C's hybrid-to-reranking pipeline isn't implemented and isn't reported as executed.

Per-query records, rankings, workload query IDs, and individual latency observations stay under ignored local reports. Committed review reports contain aggregate evidence and hashes. Bootstrap mean intervals resample normalized-query groups 2,000 times, seed 42; they describe this single baseline's query-sampling uncertainty, with no invented competing model or training-variance claim. Recall is undefined when no queries have judged E products, with explicit exclusion counts.

Basic latency uses 200 development queries selected by SHA-256 of `42|query_id`, 20 warmups once, and 3 sequential repetitions at top-100. It measures analyzer, catalog scoring, and deterministic selection, excluding HTTP/JSON/queueing. Filesystem/data caches are warm after evaluation. Percentiles from 200 samples are descriptive. The one HTTP latency observation is a smoke measurement. Neither satisfies M8's controlled concurrency/load study or M9's separately approved optimization study.

See [M3.md](reviews/M3.md) for actual quality/resource/timing evidence, failures, limitations, and understanding questions. No baseline promotion, test evaluation, M4 implementation, remote publication, or CI execution is implied.
