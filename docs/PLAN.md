> Current status, 2026-11-05. M5 supervised training and M6 returned-model/saved-ranking checks are verified. The M7 real-model HTTP gate passed on DSMLP with exact offline/HTTP features and predictions. Docker fixture checks passed separately. The user accepted M7 on 2026-11-05. The M8 CPU reference benchmark completed and passed independent evidence checks; lead acceptance and M9 study approval remain pending. See [the M8 review](reviews/M8.md). See [the M7 review](reviews/M7.md). Earlier status sections below are historical records. Source publication to the user-designated Atlas repository is authorized. Data, models and generated artifacts remain private, and GitHub Actions remains manual-only.

# AtlasRank planning proposal

Status: v0.5 records approval of v0.2 D1–D9 and amended D10. M3 is committed as 2409da6. M4 full-catalog encoding, native Linux FAISS and development-only dense/hybrid A/B are accepted; see [M4.cpu-run.md](reviews/M4.cpu-run.md). M5 experiment preparation is authorized; training and the GPU pilot await a reviewed configuration and destination. Preserve the Mac failure and all previous evidence. Remotes, publication and GitHub Actions remain unauthorized; full Linux CI remains deferred.

## Recommendation

AtlasRank should add defensible evidence for Applied ML, ML Systems, Search/Ranking, and ML Platform roles beyond the owner's production recommendation, distributed-training, and observability experience. Require supervised neural retrieval, measured inference engineering, and an explainable release process for portfolio completion.

Keep one local service and reproducible offline pipelines. Verify BM25, pretrained dense retrieval, and hybrid baselines, then run supervised PyTorch dual-encoder fine-tuning on bounded Colab GPUs before learned ranking. Profile serving before choosing a required inference optimization study. Negative results count as completed experiments when their design, execution, and limitations are reviewed.

The [initial proposal](INITIAL_PROPOSAL.md) is historical context; [Reasoning.md](Reasoning.md) supplies the review rationale. These documents define the revised scope and acceptance criteria:

- [Product requirements](PRD.md)
- [Architecture](ARCHITECTURE.md)
- [Evaluation contract](EVALUATION.md)
- [Milestones](ROADMAP.md)
- [Proposed agent instructions](AGENTS.proposed.md)

D1–D9 and amended D10 are approved. README documents the current foundation; root `AGENTS.md` adopts the approved D8 rules. `AGENTS.proposed.md` retains their planning lineage. The initially referenced `RTK.md` and `PROCESS.md` are absent; no rules were inferred or broken includes retained.

## Feasibility and risks

| Risk | Consequence | Proposed response |
| --- | --- | --- |
| ESCI judgments cover selected candidates | Catalog retrieval metrics have incomplete ground truth | Separate labeled-candidate ranking, catalog retrieval, and end-to-end reranking; report judgment coverage and known-exact Recall@20/@100 |
| No behavioral or transaction data in scope | Personalization, conversion lift, and online drift claims lack evidence | Evaluate query relevance; label load and replay experiments explicitly |
| Missing structured price and inventory fields in the published product schema | The README's price-constrained example can't guarantee a budget filter | Return text relevance results; exclude price and stock guarantees |
| Local RAM, thermal limits, and embedding cost | Full-catalog builds may exceed practical limits | Audit hardware and corpus first; batch encoding and measure peak memory |
| Mac, Colab, and Linux differences | Package support and numerical behavior may differ | Pin compatible dependencies; CPU reference path; platform smoke checks |
| Selected-candidate training differs from retrieved candidates | Ranker can fail on online candidate distribution | Compare all 3 evaluation tracks and feature distributions; don't assign negative labels to unjudged products |
| Too many services and tools | Infrastructure consumes modeling time | One process, local artifacts, no mandatory dashboard stack |
| Test reuse and pretrained-data contamination | Improvements can be overstated | Lock test access; record model provenance and contamination uncertainty |
| Early GPU work or optimization expands scope | Colab interruptions or sweeps consume the project budget | Approve sample/step/time limits before runs; persist resumable checkpoints; study one profiled bottleneck |

The [official ESCI repository](https://github.com/amazon-science/esci-data) defines a candidate-ranking task and supplies product text. AtlasRank's catalog retrieval evaluation is a project extension. Audit actual files before asserting dataset counts or usable field coverage.

## MVP and portfolio completion

The smallest credible MVP includes one English catalog snapshot, BM25, a frozen pretrained encoder with exact FAISS search, RRF, and one XGBoost ranker. Require all 3 development evaluation tracks, exact offline/online feature parity, a FastAPI endpoint, artifact validation, observability, local MLflow, Docker, CI, baseline profiling, and reproducible benchmark evidence. M8 in the [roadmap](ROADMAP.md) defines its verifiable exit. MVP completion uses development data and never opens the test set.

Portfolio completion additionally requires a substantive supervised PyTorch dual-encoder experiment (M5), a measured inference optimization study (M9), a controlled previous-bundle rollback, one approved frozen final evaluation, a technical case study, a required minimal model-comparison demo, and the milestone learning/review packets (M10). Training must change encoder weights on approved training data, produce persistent resumable checkpoints, and compare with the same pretrained initialization; a smoke test alone doesn't qualify. Neither training nor optimization must win to satisfy the evidence requirement; promotion remains separately gated.

The required minimal portfolio demo must display actual rankings, versions, request timings, available development judgments, and saved benchmark evidence. Limit it to one local comparison view with two selected pipelines, top-10 products, bundle IDs, measured request/stage timings, and available development judgments or explicit unknown labels. Read from the API or saved reports; no accounts, dashboards, deployment, or duplicated ML logic. It remains optional for MVP completion.

Select an optimization only after profiling, for example reranking depth, inference batching, or precision. Defer cross-encoders, ANN changes, Redis, Qdrant, Spark, personalization, dashboard stacks, shadow deployments, and automatic rollback until measured evidence and separate approval justify them. Kubernetes, Kafka, distributed GPU infrastructure, and unrelated features remain out of scope under the same rule.

## Conflicts resolved by this revision

- Earlier training needs experiment tracking earlier: local MLflow setup moves to M1, before the M5 Colab experiment.
- Earlier training and a CPU-capable MVP have different completion needs: M5 follows baseline verification by default. The lead can authorize a pretrained-only MVP path if GPU work is blocked; M5 remains mandatory for portfolio completion.
- An MVP test run would expose labels before later optimization: remove that path. M8 is development-only; M10 is the single approved frozen final evaluation across the declared suite.
- Optimizations can change candidate budgets: preserve the reference pipeline and register each study variant before measurement. Include retained variants in the final freeze; don't silently replace evaluation defaults.
- Exact feature parity is stronger than approximate prediction parity: enforce exact input-vector checks on the pinned CPU reference path; document cross-platform numerical differences separately.

## Approved decisions and remaining execution gates

| ID | Approved decision | Tradeoff |
| --- | --- | --- |
| D1 | US locale, `small_version == 1` judgments; all US products in the pinned metadata snapshot | Smaller modeling scope; no multilingual or leaderboard-equivalent claim |
| D2 | Official test held out; normalized-query groups split official train approximately 80/20 | Protects generalization; overlap quarantine changes usable training counts |
| D3 | Evaluation v0.2: preserve splits, gains, uncertainty, and test isolation; add explicit end-to-end comparisons | More realistic reranking evidence; incomplete judgments remain a limitation |
| D4 | Compact MiniLM baseline, exact FAISS, BM25, RRF, CPU XGBoost | Modest initial complexity; exact search may later limit throughput |
| D5 | One modular FastAPI service and immutable local artifact bundles | Easy local review; doesn't demonstrate independent service scaling |
| D6 | M5 shared-weight PyTorch/Colab dual encoder after M4; 1-hour pilot cap; provisional 6-hour total pending pilot evidence and separate approval; M8 MVP and M10 portfolio exits | Earlier modeling evidence; objective/sample/schedule and persistent storage need separate execution approval |
| D7 | Local MLflow from M1, fixture-only CI, exact feature parity, mandatory M9 optimization study | More experiment and systems evidence; full-data/GPU checks remain explicit |
| D8 | Adopt revised agent rules and require learning/review packets at every milestone | Human understanding is an exit gate; approved rules are adopted in root AGENTS.md |
| D9 | Only one approved frozen final test evaluation at M10 | Protects later training/optimization; MVP reports use development data |
| D10 | Required minimal comparison demo, measured rollback, and technical case study | Clear presentation and release evidence without frontend scope growth |

D1–D9 and amended D10 are approved; the evaluation contract remains v0.3. M4 is accepted as the frozen pretrained retrieval baseline. M5 experiment design/preparation is authorized by the current lead instruction; a separate approval of the pilot configuration and persistent checkpoint destination is required before any GPU execution. No remote, publication or deployment is authorized. The [roadmap](ROADMAP.md) defines the learning/review packet required at every milestone.

## Early correctness gates

M2 must validate the pinned XGBoost `rank:ndcg` configuration and linear gains using fixed synthetic predictions and hand-calculated NDCG, without fitting a model. Cover E/S/C/I ratios, query grouping, cutoff, ties, and zero-IDCG handling; document library edge-case differences. M2 also establishes synthetic query-specific admissibility fixtures: only train-query E positives and explicitly judged I negatives; exclude S/C, unknown and held-out pairs, cross-query assumptions, conflicting duplicates, and invalid groups. M5 tests the actual sampler and contrastive loss against these fixtures locally before the approved GPU pilot. These gates do not authorize M2 or training during M1.

## M1 evidence

The [M1 engineering review](reviews/M1.md) retains deferred Linux CI evidence. The [M2 review](reviews/M2.md) records the approved local data-contract foundation. M3 is committed as 2409da6. M4 is accepted; M5 preparation can proceed, with training held until the exact pilot is approved.
