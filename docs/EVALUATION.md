# Proposed evaluation contract

Status: v0.3 records approval of the v0.2 evaluation philosophy and the requested clarifications. Only M1 is authorized; M2 and model execution require further approval. Preserve v0.1's corpus, splits, gains, edge cases, bootstrap settings, and reference candidate budgets. This revision makes end-to-end comparison, exact feature parity, supervised training, optimization controls, and a single final test evaluation explicit. Approved reference settings remain fixed; absolute performance ceilings and the post-pilot cumulative GPU budget require later approval. Changes require a versioned amendment and comparison-impact note.

## Dataset and corpus

Use official ESCI examples filtered to `small_version == 1` and `product_locale == "us"`. The [official repository](https://github.com/amazon-science/esci-data) provides train/test assignments and joins product metadata using locale and product ID. Pin the repository revision and file SHA-256 hashes; audit actual counts, schema, missing text, duplicate pairs, conflicting labels, and product joins.

The retrieval corpus is every unique US product in that pinned product metadata file, including products associated with held-out queries. This models a fixed, available catalog and unseen queries. It doesn't measure unseen-product generalization. Never restrict the corpus to products relevant to the evaluated queries. Catalog text can build indexes; held-out labels can't fit models or choose configurations.

Missing text becomes an empty field with a reported count; retain the product. Missing joins or conflicting duplicate labels fail validation pending review. Collapse identical duplicate judgments and count them. Preserve label values and raw files. Record dataset/model license notices and attribution with provenance; review redistribution terms before publishing artifacts.

## Split policy and test access

1. Keep the official test assignment. No random pair-level split and no reallocation of test rows.
2. On official train, group by query identity and normalized query text within locale. Normalize with Unicode NFKC, casefolding, trimming, and whitespace collapse. Merge groups connected by either identity or normalized text.
3. For overlap auditing only, inspect test query identities/text. Quarantine entire training groups that overlap test identities or normalized text; preserve test rows. Record exclusions without inspecting test quality metrics.
4. Assign the remaining groups by SHA-256 of `42|locale|canonical_group_key`, using the smallest normalized text as the canonical key. Hashes in the bottom 20% of the 256-bit range go to development; the remainder go to training. Persist exact membership. These are approximate 80/20 query-group proportions.
5. Use development for early stopping and all model/configuration choices. Keep test judgments outside training, routine notebooks, and CI fixtures. Code boundaries plus a run ledger provide procedural protection; local file separation isn't a security guarantee.
6. At M10, after all model and optimization choices, freeze model/bundle hashes, candidate budgets, preprocessing, metric version, comparisons, promotion thresholds, and the final execution manifest. Require explicit technical lead test-run approval. Evaluate the entire predeclared suite in one frozen final evaluation and retain all outcomes.
7. MVP acceptance, milestone reports, optimization, and demo selection use development data only. There is no interim MVP test run. Once final test results are visible, don't tune or replace models, thresholds, or configurations; report failures against the frozen criteria. Record technical execution failures; resume only missing work with the identical manifest and lead approval, never selectively rerun unfavorable results.

The final suite includes BM25, pretrained and fine-tuned dense/hybrid paths, their matched rankers, and predeclared optimization reference/variants, even when their development results are negative. Declare the exact included runs before accessing test judgments; retain all other rejected trials in development reports. The freeze is one evaluation campaign across the suite, not one selected winning model.

Queries with normalized equivalents in test may remain separate official query IDs, so report uncertainty using normalized-query groups as resampling units. Near-duplicate paraphrases and product variants remain possible; audit samples and record residual leakage risk. A later cold-product or temporal benchmark needs a separate contract; this snapshot supports no temporal generalization claim.

## Three evaluation tracks

**A. Labeled-candidate ranking:** score every supplied judged product for each query, with no retrieval truncation. Compare BM25 scoring, pretrained/fine-tuned dense similarity, RRF over those same candidates, and each encoder's XGBoost reranker. The label pool is supplied by the dataset; results measure conditional ranking quality and diagnose behavior independently of retrieval.

**B. Catalog retrieval:** retrieve from the entire frozen corpus. Return 100 candidates from each retriever; report standalone BM25 and pretrained/fine-tuned dense results. Fuse each lexical/dense pair's deduplicated union with `RRF(d) = sum(1 / (60 + rank(d)))`, with ranks starting at 1 and absent candidates contributing 0. Keep the top 100 fused products. Compare retrievers before learned reranking using known-exact Recall@20/@100 and judgment coverage.

**C. End-to-end retrieval-to-reranking:** run the actual serving pipeline from query to top 10, without supplying judged candidates or injecting missed positives. For each encoder, compare the unchanged top-100 RRF candidate pool before and after its matched XGBoost ranker. Save candidate identities and output order; assert that reranking preserves candidate membership. Also compare the complete pretrained and fine-tuned pipelines, explicitly identifying that their candidate pools can differ. Candidate-set size variants belong to the registered optimization study and have separate IDs.

Capture the internal final ranked list, up to 100 products, before the API's top-10 truncation. C's primary metric uses the returned top 10; its Recall@20/@100 and coverage@100 diagnostics use that recorded internal list. Don't compute Recall@100 on a top-10 response or silently restore candidates discarded by an optimization variant.

Unjudged products remain unknown in B/C. For C's `pooled_ndcg@10`, assign them zero observed gain solely for metric calculation and disclose incomplete-judgment bias. Compute IDCG from all judged products for that query, including judged positives missed by retrieval; never recompute it from retrieved candidates. Never remove unknown products and compress the ranking. Report judgment coverage alongside every pooled score, and known-exact Recall@20/@100 before/after reranking. C's pooled result isn't equivalent to A or a fully judged catalog metric; don't claim that an increase establishes unrestricted catalog relevance improvement.

## Metrics and edge cases

| Measure | Frozen definition |
| --- | --- |
| Primary A: NDCG@10 | Gains E=1, S=0.1, C=0.01, I=0; direct linear gain divided by `log2(rank+1)`; IDCG from all judged products for the query |
| Secondary A: MRR@10 | Reciprocal of first E rank within 10; 0 if absent |
| Primary B: known-exact Recall@20/@100 | Retrieved judged E products divided by all judged E products for the query |
| B/C judgment coverage@10/@100 | Fraction of returned positions with any ESCI judgment; also report number returned |
| Primary C: pooled NDCG@10 | Same gain/discount and full judged-pool IDCG; unknown results have zero observed gain |
| Secondary C: known-exact Recall@20/@100 | Same known-E denominator as B, using the final reordered list; Recall@100 is invariant for a fixed 100-candidate pool |

The gain choices match the [organizers' published ESCI task description](https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search). This English-only, query-deduplicated protocol and cutoff don't imply leaderboard comparability.

Macro-average per official query ID, with equal query weights. NDCG is 0 for zero-IDCG queries; include and count them. Recall is undefined for queries with no judged E: exclude from recall's mean and publish its denominator and exclusion count. MRR remains 0 for those queries. Empty rankings score 0; shorter lists aren't padded with hits. Reject duplicate or unknown product IDs. Resolve score ties by ascending product key, never label or input order.

Compute paired deltas and 95% percentile bootstrap intervals with 2,000 resamples of normalized-query groups, seed 42. Report group/query counts. Intervals describe query sampling uncertainty, not training variability. Add development slices for whitespace-token query length (1, 2–3, 4+), title-token overlap with judged E products (zero/nonzero), and missing product titles; report slice denominators. Don't claim unsupported product categories or language coverage.

## Baselines, training, and fairness

Freeze these comparisons: BM25; frozen MiniLM cosine retrieval; BM25 plus MiniLM RRF; identical hybrid candidates plus XGBoost. Portfolio completion also requires fine-tuned versus pretrained versions of the same encoder, with matched hybrid/ranker comparisons in A/B/C. M5 trains immediately after M4 verifies retrieval baselines; M6 fits encoder-specific rankers. No result or win is assumed.

Train the ranker on complete judged training candidate groups. Features: BM25 score, dense cosine, title/query token overlap, and text lengths. Compute features using the same frozen corpus and shared serving implementation. Never use dataset row order, label-derived features, query/product IDs as predictors, split flags, or held-out statistics. RRF is the ordering baseline, not a feature computed differently on labeled and online pools.

Use query-grouped XGBoost `rank:ndcg`, integer labels E=100, S=10, C=1, I=0, and `ndcg_exp_gain=false`. This preserves the linear gain ratios. Verify library NDCG against hand-calculated fixtures; don't rely on library defaults. [XGBoost documents the gain switch](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-to-rank-task-parameters).

## Required supervised dual-encoder experiment

M5 must perform substantive supervised PyTorch fine-tuning on Google Colab GPUs, with inspectable loss, optimizer, gradient flow, and updated encoder weights. A tiny overfit/smoke test validates code but doesn't fulfill portfolio completion. Before execution, approve the actual training-query sample, pair count, objective, maximum steps/epochs, stopping rule, and minimum completed training schedule; choose these using train-only statistics and the bounded pilot, never test results.

Proposed objective: per-query contrastive softmax over judged E positives and judged I negatives with shared query/product encoder weights. Record temperature and positive sampling. Exclude S/C from this objective; count exclusions and queries without usable E/I pairs. Mask other known E products from negatives. Don't use cross-query in-batch negatives without query-specific I judgments, mine with held-out labels, or assume random/unjudged catalog products are irrelevant. Explain candidate-selection bias and incomplete supervision.

Maintain a pilot ceiling of 1 GPU-hour. The 6-GPU-hour overall M5 estimate is provisional, not an approved execution cap or allocation. Use pilot throughput, re-encoding cost, restart risk, and storage evidence to propose a cumulative budget for separate approval, counting pilot, restarts, repetitions, and GPU catalog re-encoding. No GPU execution is authorized in M1. The pilot establishes throughput; obtain approval of the substantive schedule before continuing. Start with one training configuration, seed 42. Add seeds 43/44 within the approved budget when feasible; otherwise disclose exploratory single-seed evidence. If no substantive schedule fits, request a revised scope/budget and leave M5 incomplete.

Persist versioned checkpoints outside Colab's temporary runtime, including model/tokenizer, optimizer/scheduler state, random-number states, sampler/data-order progress, global step, configuration, and provenance. Verify restored training progress and resumed updates; never restart the run ledger to bypass caps. Record GPU type, software, precision, actual steps/time, interruptions, and checkpoint hashes. Import/export MLflow run records without relying on an always-on server. Rebuild full-catalog product embeddings and indexes per encoder; compare pretrained/fine-tuned retrieval on identical manifests with paired uncertainty and failure analysis.

Proposed non-neural tuning ceilings: 4 predeclared BM25 configurations and 8 XGBoost model fits in total across encoder variants, selected on development only; one pretrained initialization, one text representation, and fixed reference RRF settings. Register configurations before each sweep. Additional neural configurations or trials need a separate approved experiment plan and budget. Use seed 42 for a released stochastic model, rather than selecting the best seed. Negative results remain in the run ledger.

Freeze model revisions, text/tokenization, corpus, split, candidate budgets, tuning trial budget, and hardware per comparison. Preserve configs, checkpoints, per-query outputs, MLflow run IDs, failures, and package versions. PyTorch cautions that [reproducibility across releases and platforms isn't guaranteed](https://docs.pytorch.org/docs/stable/notes/randomness.html); declare numerical tolerances and backend differences.

## Exact offline/online feature parity

In CI, execute the offline evaluation path and the actual HTTP serving path with the same immutable fixture bundle, query/product pairs, candidate set, CPU backend, locked environment, and deterministic preprocessing/encoding policy. Capture each path's actual feature matrix immediately before ranker prediction. Join comparisons by product key and canonical row order; require identical feature names/order, dtypes, missing-value masks, and exact values with zero tolerance. Any mismatch fails CI. A direct helper-versus-itself comparison is insufficient.

Fixtures cover lexical and dense scores, Unicode/whitespace normalization, title-token overlap, empty/missing text, ties, permuted candidates, different batch/request sizes, and save/load. Include an intentional order or normalization defect to prove the check detects a mismatch. At M6, also run recorded reference-CPU integration parity checks with the actual pinned pretrained/trained bundles. A lead-approved pretrained-only MVP defers only the trained-bundle check until M5 completes; it never waives parity for an available bundle. Routine CI uses a small deterministic fixture encoder with no model downloads. Reject incompatible schemas before readiness. Approximate cross-platform prediction checks are separate and can't relax feature parity on the reference path.

## Serving benchmark

Use a hash-selected, fixed development-query workload, a separate load generator, CPU serving, one worker, and no response cache. Record hardware/RAM, OS, container/native execution, dependencies, power conditions, threads, query lengths, bundle, candidate counts, and workload hash. Don't compare Mac native and Linux container measurements as the same environment.

Warm up with 100 requests; then run 1,000 measured requests at concurrency 1, 4, and 8, repeating each setting 3 times. Report client end-to-end p50/p95/p99, server stage timings, successful requests/s, errors/timeouts, peak RSS, index size, and startup/readiness time. Publish per-repeat results; p99 from small samples is descriptive. Count timeouts as failures and report their threshold. Measure cold startup separately. Stop on resource pressure and report the interruption.

## Required inference optimization study

M8 profiles encoding, lexical/vector retrieval, fusion, feature construction, reranking, queueing, and end-to-end latency. Before M9, use that evidence to propose one bottleneck hypothesis, alternative designs, a bounded variant set, and adoption criteria. Approve the study and absolute latency/memory/error limits before running variants. A reference profile alone doesn't satisfy portfolio completion.

For example, if reranking dominates, retain retrieval's 100-per-retriever and RRF-60 defaults, and compare reranking the top 20/50/100 fused candidates, returning the top 10 from the selected subset. Keep the 100-candidate pipeline as the reference and assign separate IDs to each depth variant. If another stage dominates, propose a different intervention. Any changed encoding, precision, index, cache, or candidate policy needs a versioned variant manifest and fresh compatibility/parity checks.

Use the same development queries, corpus, hardware, process/thread limits, load generator, warmup, repetitions, timeouts, and measurement boundaries as the reference. Change one factor at a time, interleave reference/variant runs to reduce thermal/order bias, and record actual temperatures/power state when available. Report C quality and B known-exact Recall@20/@100, judgment coverage, p50/p95/p99, throughput, errors, memory, and concurrency failures. Fixed-candidate reranking studies must also show invariant candidate membership at the retrieval boundary. Separate closed-loop concurrency results from any explicitly added arrival-rate load test.

Publish per-variant paired quality deltas, all repeat-level performance results, and an adoption or rejection rationale. The lead approves a concrete minimum worthwhile latency/throughput improvement after baseline profiling and before the variant study. Keep the baseline if gains don't justify quality loss, resource use, or complexity. A fully executed negative study qualifies; it never justifies claiming an improvement. Register retained variants and the chosen serving configuration before final test freeze. Preserve rejected configurations in development reports.

## Promotion and final evidence

Before evaluating a candidate release, the lead approves absolute p95 latency, memory, timeout, and error ceilings from baseline measurements. Preserve the proposed relative gates: lower bounds of paired 95% development deltas at least -0.005 for both A NDCG@10 and C pooled NDCG@10; B known-exact Recall@100 loss no more than 0.01; p95 at most 1.10 times the approved incumbent under identical workloads; zero unexpected errors in the fixed smoke workload. All are absolute score deltas except the latency ratio. Adding C prevents promotion based on labeled candidates alone. Hold promotion until absolute ceilings are filled in; the optimization study also needs its preapproved worthwhile-improvement criterion.

Claim a ranking improvement only with a positive paired interval on the declared track-specific primary metric; qualify C claims as pooled-judgment evidence and disclose multiple-comparison and seed limitations. A model can complete an experiment and fail promotion. All 3 tracks, training evidence, optimization evidence, and failed comparisons belong in the final engineering review. Use the single approved frozen test evaluation for reporting only; never subsequent tuning. Controlled failure and rollback exercises are replay tests; traffic metrics alone don't establish relevance drift or online business impact.

## Early correctness gates before modeling

M2 must run a pinned-library, zero-update XGBoost conformance check on synthetic grouped labels E=100/S=10/C=1/I=0 and fixed predictions. Inspect `rank:ndcg`, disable exponential gain explicitly, and compare library NDCG with independent hand calculations using E=1/S=0.1/C=0.01/I=0. Include multiple query groups, cutoff, ties, nontrivial permutations, and zero-IDCG cases. Default exponential gain must be detected as incompatible; adapt documented library zero-IDCG behavior to the approved evaluator rather than silently changing the contract. No model fitting is needed for this gate; gradient/objective update sanity checks belong to separately authorized M6 modeling.

M2 establishes synthetic supervision truth tables. M5 tests the actual sampler and loss against them locally before the Colab pilot: E is positive; only query-specific judged I from the training manifest may be negative; S/C, unknown pairs, other-query judgments, and held-out groups are excluded. Test multiple E products, duplicate/conflicting judgments, absent E/I, deterministic seeds, finite loss, and negative masks. Verify forbidden candidates receive zero contribution, admitted negatives contribute, and no positive is penalized as a negative. Fail closed on invalid groups and report exclusions. No sampler or metric implementation is authorized during M1.
