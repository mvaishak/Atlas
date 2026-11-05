# AtlasRank

AtlasRank is a product-search project built on the English Amazon ESCI dataset. It compares lexical retrieval, a supervised PyTorch dual encoder and a learned reranker, then checks that offline ranking and HTTP serving use the same features and predictions.

The repository is named Atlas. The Python package and application are named AtlasRank. Source, synthetic fixtures and engineering reviews are public. Catalog data, model weights, indexes, generated reports and runtime logs stay in private local storage.

## Search pipeline

BM25 and a MiniLM dual encoder each retrieve 100 products. FAISS handles dense retrieval. Reciprocal rank fusion uses `k=60`, and an optional XGBoost ranker reranks the fixed candidate pool with 5 shared features. FastAPI runs the pipeline in one process.

The API exposes `/search`, `/rerank`, readiness, liveness and aggregate metrics. Responses include pipeline and artifact versions, scores and measured stage timings. BM25 is the default. The M6 XGBoost pipeline remains experimental.

## Results so far

All relevance results use development judgments. Official test judgments remain locked, and unjudged products remain unknown.

| Experiment | Observed result |
| --- | --- |
| M5 supervised dual-encoder training | 574 optimizer updates over 9,180 training groups. Returned checkpoint and exported-model evidence verified. |
| M5 hybrid retrieval | Known-exact Recall@100 of 0.493986, a +0.005055 difference from M4. Paired 95% interval [+0.001134, +0.008991]. |
| M6 judged-candidate ranking, Track A | NDCG@10 of 0.711080, compared with 0.704503 for the frozen hybrid baseline. Paired difference interval [+0.003795, +0.009266]. |
| M6 retrieved-pool reranking, Track C | Observed NDCG@10 fell from 0.247935 for RRF to 0.244565. Paired difference interval [−0.006323, −0.000306]. Fixed-pool Recall@100 was unchanged. |
| M7 real-model HTTP check | 3 queries, each with 100 search candidates and 7 rerank candidates. All 6 offline/HTTP capture pairs matched exactly. |

M5's small positive hybrid result has pipeline limitations. Historical padding, normalization and FAISS tie handling differ between M4 and M5, and standalone dense improvement remains unestablished. Full historical M5 retrieval artifacts are stored externally. [M5 evidence and limitations](docs/reviews/M5.5.md)

M6's returned model and saved rankings passed independent hash, metric and paired-interval checks across 4,129 development queries. Some original execution records remain missing. Its negative pooled Track C result keeps model promotion pending. [M6 return verification](docs/reviews/M6.return.md), [M6 results](docs/reviews/M6.md)

The M7 check used the frozen M4 pretrained encoder and M6 ranker in an isolated native CPU environment on DSMLP with 4 CPUs and 32 GiB RAM. It took 58.17 seconds including cleanup and reached 5.52 GiB sampled peak RSS. These are bounded correctness-smoke observations. Performance benchmarking starts in M8, which hasn't begun. [M7 review](docs/reviews/M7.md)

That run recorded the historical and runtime kernel/glibc identities separately under an explicit compatibility policy. The reconstructed reference recipe matched the preserved sealed digest. Docker fixture checks passed separately. Full neural startup inside that Docker image remains unverified. The user accepted M7 on 2026-11-05. [Serving operations](docs/M7_SERVING.md), [DSMLP procedure](docs/M7_DSMLP.md)

## Set up

Use Python 3.12.11 and `uv` 0.8.22. The checked-in lock defines the development environment.

```sh
uv sync --locked --group dev --python 3.12.11
uv pip check
```

The recovered DSMLP CPU environment has separate exact pins in [its environment file](environments/dsmlp-cpu/requirements.txt). Follow the DSMLP procedure when reproducing that acceptance run.

## Run checks

These tests use synthetic fixtures and don't need the private catalog or models.

```sh
uv run --locked python -m pytest -q tests/test_m7_bundle.py tests/test_m7_api.py \
  tests/test_ranking_features.py tests/test_ranking_parity.py \
  tests/test_search_api.py tests/test_demo.py
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked mypy
```

Run native retrieval, Torch training and runner-admission suites in separate processes. The Mac Torch/FAISS OpenMP conflict remains unresolved. The [reproduction guide](docs/REPRODUCING_M3_M5.md) and [M7 review](docs/reviews/M7.md) record test groups, skipped checks and environment limits. GitHub Actions is manual-only.

The optional headless browser test requires `ATLASRANK_BROWSER_TESTS=1` and an existing Chromium installation. Routine fixture checks leave it disabled.

For a data-free HTTP smoke check, follow the Docker fixture commands in [serving operations](docs/M7_SERVING.md).

## Run with private artifacts

BM25 serving needs a compatible local index. Replace the index path with your verified bundle.

```sh
uv run --locked python -m atlasrank serve \
  --index data/esci/indexes/m3-bm25-v1 --port 8000
```

```sh
curl -sS http://127.0.0.1:8000/health/ready
curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"wireless headphones","top_k":10}'
```

Complete neural serving needs the sealed descriptor, matching catalog, encoder, index and ranker bundles, plus a compatible Linux CPU environment. It requires at least 8 GiB available RAM at admission and enforces a 12 GiB worker RSS ceiling. [Serving operations](docs/M7_SERVING.md) covers preparation, validation and launch commands. Missing or incompatible bundles keep the service unready.

Data acquisition, full-catalog encoding, training and evaluation are separate operations. Their inputs, resource budgets and evidence are recorded in the [reproduction guide](docs/REPRODUCING_M3_M5.md) and [evaluation contract](docs/EVALUATION.md).

## Code and reviews

- `src/atlasrank/data/` validates dataset and split contracts.
- `retrieval/`, `features/`, `ranking/` and `training/` hold retrieval, shared features, ranker integration and PyTorch training.
- `evaluation/` implements metrics and paired comparisons. `artifacts/` validates versioned bundles.
- `serving/` contains the API and comparison view. `tests/` contains synthetic fixtures.
- [Architecture](docs/ARCHITECTURE.md), [requirements](docs/PRD.md) and [roadmap](docs/ROADMAP.md) define the project scope.
- [Engineering reviews](docs/reviews/) record decisions, alternatives, failures, commands and evidence. [Repository rules](AGENTS.md) govern contributions and execution.
