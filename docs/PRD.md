> Current status, 2026-11-04. M5 supervised training and M6 returned-model/saved-ranking checks are verified. The M7 real-model HTTP gate passed on DSMLP with exact offline/HTTP features and predictions. Docker fixture checks passed separately. Human code approval and lead acceptance remain pending. M8 has not started. See [the M7 review](reviews/M7.md). Earlier status sections below are historical records. Source publication to the user-designated Atlas repository is authorized. Data, models and generated artifacts remain private, and GitHub Actions remains manual-only.

# AtlasRank product requirements

Status: v0.5, approved product scope with required minimal portfolio demo. Owner: technical lead. M3 is committed as 2409da6. M4 full-catalog pretrained retrieval and development A/B evidence are accepted as the frozen baseline; see [M4.cpu-run.md](reviews/M4.cpu-run.md). M5 design preparation is authorized; test judgments remain locked and training requires separate pilot approval.

## Problem and users

AtlasRank helps a shopper find products from a text query. It must provide new evidence of supervised neural modeling, inference engineering, and engineering judgment for Applied ML, ML Systems, Search/Ranking, and ML Platform roles.

The product is a local, production-oriented search prototype. Its scope is query-product relevance. Business impact and personalization require additional data and experiments.

## MVP requirements

1. Prepare a versioned English ESCI catalog and query-disjoint training/development manifests. Preserve a locked test set.
2. Compare BM25, frozen dense retrieval, hybrid fusion, and a learned reranker on the same documented data.
3. Accept a query and `top_k` through an API; return unique ranked products, scores, bundle version, and stage timings. Validate empty, oversized, and unsupported requests.
4. Run labeled-candidate ranking, catalog retrieval, and end-to-end retrieval-to-reranking comparisons on development data, preserving incomplete-judgment caveats.
5. Reproduce a CPU serving run in Docker; record latency percentiles, throughput, errors, memory, and index size under a fixed workload.
6. Expose readiness, liveness, structured logs, and aggregate request/stage metrics. Reject incompatible artifacts before readiness.
7. Document setup, data provenance, experiments, limitations, and commands. Run automated tests and static checks in GitHub Actions.

## MVP exit: M8

- A clean environment can rebuild a small fixture pipeline and load a compatible artifact bundle.
- All baselines use identical manifests and metric definitions; an independent metric fixture verifies calculations.
- Offline evaluation and HTTP serving produce exactly equal ranker-input feature vectors under the pinned CPU reference configuration; CI fails on any mismatch. Predictions have separately declared tolerances.
- A versioned report includes measured quality, uncertainty, performance, and known failure cases.
- Required tests pass; the technical lead reviews code and each milestone's learning packet. Test judgments remain locked.

Improvement over a baseline isn't required to complete an experiment. Model promotion requires the separate approved gates in the [evaluation contract](EVALUATION.md).

## Portfolio exit: M10

- Complete a substantive supervised PyTorch dual-encoder experiment on approved Colab GPUs immediately after baseline retrieval verification. Record bounded configurations, changed weights, persistent resumable checkpoints, and pretrained/fine-tuned development comparisons; smoke tests alone don't qualify.
- Complete a profiling-led inference optimization study with a controlled reference/variant benchmark, quality/latency tradeoffs, and an evidence-based adoption or rejection decision.
- Demonstrate previous-bundle rollback under a controlled fault, publish one approved frozen final evaluation across all 3 tracks, and deliver the minimal comparison demo, and a technical case study plus AI-development/review log.
- Supply every milestone's architecture explanation, alternatives, failure modes, experiment evidence, interview questions, and human-understanding checks. The technical lead approves the evidence and claims.

MVP capability doesn't require trained encoder weights. The default roadmap trains early; a lead-approved pretrained-only MVP path leaves training outstanding for portfolio completion. A measured negative training or optimization result can satisfy the study requirement without passing model-promotion gates.

## Required minimal portfolio demo

At M10, provide one local comparison view for two selected retrieval/ranking pipelines: top-10 products, bundle IDs, measured request/stage timings, and development judgments or explicit unknown labels. Use the API or saved measured reports, label per-request timings separately from benchmark aggregates, and prove one reproducible comparison against recorded outputs. No accounts, dashboard suite, cloud deployment, or duplicated ML logic. Required for portfolio completion; optional for MVP.

## Out of scope

Personalization, checkout, price guarantees, live inventory, online A/B testing, multilingual support, and elaborate frontend work. Kubernetes, Kafka, distributed GPU infrastructure, and unrelated features require measured justification and separate approval. Local replay doesn't establish user impact.

## Early validation

Before learned modeling, M2 checks XGBoost linear-gain NDCG against synthetic hand calculations and fixes the query-specific supervision admissibility fixtures. M5 must verify the actual negative sampler/loss against those fixtures before Colab. The pilot is capped at 1 GPU-hour; 6 GPU-hours overall is a provisional estimate requiring pilot evidence and separate approval.
