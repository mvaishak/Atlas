# AtlasRank M7 serving reliability plan

**Status:** Implementation and technical acceptance evidence are ready for review as of 2026-11-04 Pacific. Full real-model Linux startup and HTTP parity passed on DSMLP under the explicitly opted-in platform policy. Docker fixture checks passed separately. See [M7 review](reviews/M7.md). Human code approval and lead acceptance remain pending.

Implementation details and rulings: execution plan (`docs/superpowers/plans/2026-11-04-m7-serving.md`, private evidence). Artifact locations: [directory](directory.md).  
**Predecessor:** M6 development evaluation completed  
**Objective:** Serve the frozen retrieval and ranking pipelines reliably through one CPU-based FastAPI application  
**Training required:** None  
**Full development evaluation required:** None  
**Official test access:** Prohibited

---

## 1. Objective

Turn AtlasRank's existing search and reranking components into a reproducible, validated, observable serving application.

M7 covers serving correctness, reliability and deployability. The frozen evaluation results remain the quality reference.

The implementation must reuse the existing FastAPI service, original retrieval components, shared feature implementation, and frozen model artifacts.

### Required exit criteria

M7 is complete when:

1. A complete versioned serving bundle is validated before readiness.
2. HTTP requests are strictly validated and resource-bounded.
3. Offline and HTTP ranking paths produce exactly equal feature matrices on the pinned CPU fixtures.
4. Invalid or incompatible artifacts prevent readiness.
5. Liveness, readiness, aggregate metrics, and structured logging work correctly.
6. A CPU Docker image builds and passes a serving smoke test.
7. Failure-path tests and the M7 engineering review are completed.

No new neural training, embedding generation, retrieval tuning, or official test evaluation is permitted.

## 2. Existing components to reuse

| Component | Existing evidence | M7 action |
|---|---|---|
| M3 BM25 | Frozen index verified | Reuse |
| M4 MiniLM + FAISS | Frozen catalog and encoder verified | Reuse |
| M4 RRF | Existing fusion implementation | Reuse |
| M6 XGBoost | One fit, sealed model verified | Load read-only |
| Shared ranking features | Five-feature implementation | Reuse unchanged |
| FastAPI application | Existing `/search` and `/rerank` routes | Extend |
| M6 comparison UI | Existing bounded saved comparison | Preserve |
| M6 development report | 4,129 evaluated queries | Preserve |

The existing application factory is `atlasrank.serving.app.create_app`. M6 also established separate offline and HTTP feature boundaries through `evaluation/ranking.py` and `serving/ranking.py`.

Do not create a second API service or duplicate feature-generation logic.

### Model-selection decision

The M6 model improved Track A but reduced Track C observed NDCG@10 relative to RRF.

Therefore:

- Preserve the existing BM25 search behavior.
- Support M4 RRF as a retrieval pipeline when its complete bundle is available.
- Expose M6 ranking as an explicitly selected, experimental pipeline.
- Do not promote M6 as the default production ranker solely because it beat Track A baselines.
- Never substitute a saved comparison fixture for live inference without labeling the operation.

## 3. Proposed architecture

```text
HTTP Request
     |
     v
FastAPI request validation
     |
     v
Pipeline selector
     |
     +---- BM25 ------------------------+
     |                                  |
     +---- M4 Dense --------------------+
     |                                  |
     +---- BM25 + Dense --> RRF --------+
     |                         |
     |                         v
     |                 Optional M6 reranking
     |                         |
     +-------------------------+
                               |
                               v
                    Ranked product records
                               |
                               v
                Timings + version metadata
                               |
                               v
                         HTTP Response
```

Startup is a separate operation:

```text
Configuration
     |
     v
Complete artifact manifest
     |
     v
Checksums and compatibility validation
     |
     v
Load immutable catalog/index/encoder/ranker
     |
     v
Ready to serve
```

One CPU process owns the loaded artifacts. Request-specific state must remain isolated.

## 4. API contract

Extend the existing application without breaking existing route behavior.

### `POST /search`

Proposed request fields:

| Field | Type | Rules |
|---|---|---|
| `query` | String | Nonempty after canonical normalization. Maximum 512 characters and 128 analyzer tokens |
| `top_k` | Integer | 1–100. Default 10 (preserves existing API) |
| `pipeline` | Enum | `bm25`, `dense_m4`, `rrf_m4`, `m6_xgb_m4` |
| `locale` | String | Only the validated US locale |

The existing default search behavior remains BM25 for backward compatibility. The selected pipeline must be reported explicitly.

The `m6_xgb_m4` pipeline uses the original BM25 top-100 and M4 dense top-100 retrieval, RRF-60 fusion, and the matched M6 ranker over the fused top-100 pool.

The response includes:

- Request identifier
- Selected pipeline identifier
- Artifact/bundle identifier
- Product locale, product ID, title, rank, and method-specific score
- Candidate counts
- Measured stage timings and total server time
- Clear indication of the score type

Scores are ranking scores, not calibrated relevance probabilities.

The API must reject malformed JSON, empty queries, oversized inputs, invalid `top_k`, unsupported locales, and unknown pipelines.

Unavailable configured pipelines return a clear service-unavailable error rather than silently falling back to a different model.

### Health and metrics

| Route | Expected behavior |
|---|---|
| `GET /health/live` | Process is alive. Does not require model readiness |
| `GET /health/ready` | Returns success only when the configured serving bundle is fully valid and loaded |
| `GET /metrics` | Aggregate request, error, latency, and pipeline metrics |

Readiness must fail closed on incompatible or corrupt artifacts.

Use bounded metric labels. Never include query text or product IDs in metric labels.

## 5. Serving bundle specification

The M6 sealed model is one component of the serving bundle, not the complete serving bundle.

A complete M7 bundle needs:

- Versioned catalog snapshot and product identity mapping
- Validated BM25 index
- Validated M4 FAISS index and product vectors
- Pinned MiniLM model/tokenizer and text recipe
- Optional sealed M6 XGBoost model
- Feature schema and provenance bindings
- Dependency versions and artifact checksums
- Explicit serving configuration and pipeline identifiers

The loader verifies that all components refer to compatible catalog, encoder, retrieval, and feature-schema versions.

The original M6 model weights must not be modified.

Use immutable bundle directories. No in-place replacement of active artifacts.

### Frozen M6 model

SHA-256:

`91b91e61d731c59bba4f0a3964b2aeeacae20a02aefe4bd91643da69060ea19c`

Source:

`MyDrive/AtlasRank/m6/sealed-model-v1/`

The serving bundle must reference this artifact without inventing missing original input-manifest evidence.

## 6. Proposed code changes

These paths use the existing repository structure. Before editing, check whether each responsibility already has an implementation.

| File | Intended change |
|---|---|
| `src/atlasrank/serving/app.py` | Extend the existing FastAPI application, routing, and lifecycle |
| `src/atlasrank/serving/ranking.py` | Connect verified live retrieval and reranking paths |
| `src/atlasrank/artifacts/serving_bundle.py` | New complete-bundle validation module, if no existing equivalent |
| `src/atlasrank/serving/metrics.py` | Bounded request and stage instrumentation |
| `src/atlasrank/serving/runtime.py` | Immutable loaded-runtime state, if not already implemented |
| `configs/m7-serving-v1.json` | Frozen serving configuration |
| `tests/test_m7_bundle.py` | Valid and invalid artifact cases |
| `tests/test_m7_api.py` | Request, readiness, failure, and compatibility tests |
| `tests/test_m7_parity.py` | Exact offline/HTTP feature-parity tests |
| `docker/Dockerfile.cpu` | CPU-only deployment image |
| `docs/M7_SERVING.md` | Reproduction and operation instructions |
| `docs/reviews/M7.md` | Engineering evidence and final milestone review |

Reuse existing tests and modules wherever possible instead of maintaining duplicate implementations.

## 7. Implementation sequence

### Phase 0, Repository and artifact reconciliation

Before changing application code:

- Inspect the current source tree and identify the existing serving, ranking, and bundle-loader interfaces.
- Record the actual local revision and outstanding changes.
- Reconcile stale M6 status notes in the roadmap and repository guidance with the completed external M6 evidence.
- Confirm which complete serving artifacts are available locally and which exist only in Google Drive.

**Exit:** One dependency inventory, with available locations and missing historical evidence distinguished explicitly.

### Phase 1, Bundle validation

Implement or extend the complete-bundle validator.

Cover catalog identities, model compatibility, file checksums, missing files, corrupt files, feature schemas, and version mismatches.

**Exit:** Valid fixture loads. Invalid fixture never becomes ready.

### Phase 2, Reliable HTTP serving

Connect the existing BM25, M4 retrieval, RRF, and optional M6 ranking implementations through the existing FastAPI application.

Keep the original BM25 route and M6 saved-comparison behavior backward compatible.

**Exit:** Actual HTTP output matches the corresponding offline ranking path on deterministic fixtures.

### Phase 3, Observability and safety

Add structured JSON request logging, bounded aggregate metrics, stage timings, request limits, safe error responses, and explicit readiness transitions.

Do not log raw queries, product IDs, credentials, or sensitive request payloads.

**Exit:** Success and failure requests are observable without exposing raw query content.

### Phase 4, CPU Docker deployment

Build a single-process CPU image using pinned dependencies.

First run the tiny deterministic fixture bundle. Then perform one bounded actual-bundle startup and smoke test on a compatible Linux CPU environment.

Record startup duration, peak memory, bundle identity, API responses, and shutdown behavior.

**Exit:** Docker build, fixture smoke test, and actual-bundle smoke evidence pass.

### Phase 5, Engineering review

Document architecture, design choices, test commands, observed results, limitations, failures, and unresolved issues.

**Exit:** M7 review packet accepted, with M8 profiling prerequisites identified.

## 8. Validation matrix

| Test | Required result |
|---|---|
| Correct bundle | Loads and becomes ready |
| Missing index | Readiness remains false |
| Corrupt model | Load rejected |
| Incompatible encoder/ranker | Load rejected |
| Wrong catalog identity | Load rejected |
| Empty or oversized query | Validation error |
| Duplicate candidate identities | Rejected by ranking contract |
| Empty retrieval results | Valid empty result without crash |
| Repeated requests | Consistent results and isolated state |
| Offline versus HTTP features | Exact names, order, dtypes, masks, identities, and values |
| Artificial feature mismatch | Test fails deliberately |
| Model serialization | Prediction contract preserved |
| Structured logs | No raw query text |
| Metrics | Bounded labels |
| Docker | Starts, becomes ready, serves, and stops cleanly |

The full original development evaluation is not rerun for M7. The saved M6 report is reference evidence, not a replacement for actual serving-contract tests.

## 9. Design alternatives

**Alternative A, Separate services for BM25, dense retrieval, and ranking**

Rejected. This adds networking, deployment, and observability complexity without a demonstrated need at the current scale.

**Alternative B, Reuse one modular FastAPI process**

Selected. It is consistent with AtlasRank's approved architecture and simplifies request tracing, bundle compatibility, and local CPU deployment.

**Alternative C, Serve directly from Colab**

Rejected as the deployment design. Colab remains useful for bounded Linux CPU checks but is not a durable serving environment.

## 10. Resource and execution policy

M7 should require:

- Zero GPU training hours
- Zero new XGBoost fits
- Zero catalog re-encoding
- Zero full development evaluation reruns
- No external deployment or publication
- No official test-judgment access

The full catalog runtime has already required substantial memory during M6. Do not assume the smaller BM25-only serving budget will accommodate the full neural bundle.

Validate available memory before full startup. Record actual peak RSS. Stop on resource pressure rather than silently shrinking the catalog or changing the model.

Docker smoke tests are not substitutes for the controlled latency and throughput experiment planned for M8.

## 11. Risks and mitigations

| Risk | Response |
|---|---|
| Native Mac Torch/FAISS conflict | Use isolated Linux CPU integration. Preserve Mac failure evidence |
| Incompatible recovered artifacts | Validate complete provenance before readiness |
| M6 ranker underperforms RRF | Keep M6 experimental. No default promotion |
| Excessive process memory | Bounded startup and explicit resource checks |
| Request-time model reloads | Load once during application startup |
| Offline/HTTP feature differences | Exact boundary-level parity tests |
| Unjudged products mistaken for negatives | No judgment-dependent live filtering |
| Colab reset | Keep original bundles and recovery scripts persistent |
| Excessive project scope | Defer benchmarking, optimization, and full demo work to M8–M10 |

## 12. Review and human-understanding requirements

The M7 engineering review must include actual commands and observed evidence, not just proposed tests.

The technical lead should be able to explain:

1. Why model readiness is different from process liveness.
2. How artifact provenance prevents incompatible model/index combinations.
3. Why exact feature parity matters more than approximate output-score agreement.
4. How concurrent HTTP requests share immutable model state safely.
5. Why a ranker can complete evaluation successfully without being promoted to the default serving pipeline.

Record AI-generated changes, human corrections, accepted decisions, and unresolved issues.

## 13. M7 completion decision

**M7 is accepted only after all critical bundle, API, feature-parity, and Docker evidence is verified.**

M6's negative Track C result remains part of the record. M7 does not alter it.

After lead acceptance and separate execution approval, M8 can measure controlled CPU serving performance baselines and stage profiles.

---

**Original proposal status:** The original proposal preceded implementation. See the dated execution plan and M7 review for actual commands, results and outstanding gates.

## Execution refinements, 2026-11-04

- Preserve existing API limits and the default BM25 behavior. M6 remains explicitly experimental.
- Seal references to immutable artifacts. Never copy data into the image. Pin the approved CPU query recipe separately from the CUDA product recipe, including runtime provenance.
- Reject oversized request bodies above 16 KiB before JSON parsing. Use one shared nonblocking admission lock for complete-runtime search and fixed-candidate reranking.
- Emit bounded aggregate metrics using the existing stack. Per-request stage times remain separate from M8 benchmark percentiles.
- Complete-bundle startup failure disables every configured pipeline. Memory admission failure removes readiness until restart. Preserve legacy optional-ranker semantics outside complete-bundle mode.
- Local BM25-only fixture smoke uses a 256 MiB available-memory admission minimum. Full neural startup requires 8 GiB available memory and a 12 GiB process ceiling, plus external resource enforcement.
- Docker was initially unreachable. CLI discovery subsequently reached the existing daemon. The pinned CPU image, lexical container smoke and native sealed-model load passed. Full neural startup in that image remained unrun because the VM has 7.65 GiB total RAM, below the 8 GiB available-memory gate. The later isolated DSMLP environment passed actual-model startup/HTTP parity, as recorded in the current review.

## Source publication and acceptance boundary

The public checkout contains source, deterministic fixtures, serving procedures and pinned environment recipes. Full real-model reproduction requires separately supplied private immutable components and provenance. Generated validation JSON, raw HTTP captures and model/data artifacts remain private, with locations and limitations recorded in [directory.md](directory.md). Source publication authorization is separate from human code approval, lead milestone acceptance and M8 execution.
