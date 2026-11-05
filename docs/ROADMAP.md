> Current status, 2026-11-04. M5 supervised training and M6 returned-model/saved-ranking checks are verified. The M7 real-model HTTP gate passed on DSMLP with exact offline/HTTP features and predictions. Docker fixture checks passed separately. Human code approval and lead acceptance remain pending. M8 has not started. See [the M7 review](reviews/M7.md). Earlier status sections below are historical records. Source publication to the user-designated Atlas repository is authorized. Data, models and generated artifacts remain private, and GitHub Actions remains manual-only.

# Proposed implementation milestones

M5 latest: [M5.5 closeout](reviews/M5.5.md) consolidates completed training and the user's
manual trained-catalog development comparisons. [M5 status](reviews/M5.md) and the
[reproduction guide](REPRODUCING_M3_M5.md) distinguish independently verified local
weights/statistical replay from external full M5 chunks/index/rankings. No training,
full encoding or full retrieval rerun occurred during closeout. Historical padding,
normalization and dense tie-handling differences are recorded. M5 hybrid development
Recall@100 has a small positive paired interval; standalone dense improvement is not
established. The subsequently approved bounded M6 demo slice is approved for local commit;
see [M6.demo.md](reviews/M6.demo.md). It reuses the existing app for live BM25 and
five saved M3/M4 comparisons. Learned-ranker work and model promotion remain
unauthorized; this does not complete the full M6 exit requirements.

M6 feature/parity first phase is implemented for review: [M6.features.md](reviews/M6.features.md).
The five approved features and real offline/HTTP deterministic-predictor boundary pass
exact fixtures. Native XGBoost prediction failed in Mac libomp and remains a separate
Linux gate. No fitting, actual-bundle integration or full learned M6 completion is claimed.

Status: v0.8 records M5.5 source/evidence consolidation. M3 remains committed as 2409da6;
M4 pretrained results remain frozen. Preserve all historical experiments and manifests,
the native Linux bundle and the unresolved Mac OpenMP failure. Full Linux CI and MPS
remain unrun. EVALUATION.md v0.3 is unchanged; test judgments remain locked.

| Milestone | Scope | Independently verifiable exit |
| --- | --- | --- |
| M0: Plan review | Record approved D1–D9 and amended D10; evaluation v0.3 | Written decisions, completion criteria, and selected next milestone |
| M1: Foundation and tracking | Hardware/package compatibility audit; minimal package; pytest, Ruff, mypy, GitHub Actions; dependency lock; local MLflow | Tiny CPU fixture check and MLflow record/export round trip pass; Mac/Linux core-package import checks; RAM/disk and tracking paths documented |
| M2: Data contract | Acquisition instructions, provenance, catalog checks, overlap audit, frozen split manifests; metric implementation | Repeated preparation gives identical hashes; hand-calculated metric tests plus pinned XGBoost zero-update linear-gain checks and query-specific negative-admissibility fixtures; join and leakage checks pass |
| M3: Lexical vertical slice | BM25 build/load, offline search, preliminary API and bundle schema | Tiny golden corpus produces expected retrieval; save/load parity; request validation and readiness tests; development baseline report |
| M4: Verified dense and hybrid baselines | Pinned pretrained encoder, batched product encoding, exact FAISS, RRF | Cosine search matches brute force on fixtures; normalization/identity tests; A/B development reports and C's pre-ranker reference; measured build memory |
| M5: Supervised neural retrieval | Approved bounded Colab PyTorch pilot and substantive dual-encoder fine-tuning; persistent checkpoints | Actual sampler/loss passes M2 supervision truth tables locally before the GPU pilot; verified gradients/weight changes and resume; completed approved schedule beyond smoke test; train-only sampling audit; full-corpus pretrained/fine-tuned A/B comparisons, matching indexes, run ledger, checkpoints, failures, and uncertainty |
| M5.5: Reproducibility closeout | Consolidate completed M3–M5 source and evidence, without expensive reruns | Commands, pinned model/catalog, historical format adapter, lightweight statistical replay, synthetic end-to-end tests, explicit external-artifact gaps; see review |
| M6: Learned ranking and parity | Shared features, CPU query-grouped XGBoost per encoder, thin serving route for integration checks | Label/split exclusions; exact offline/HTTP feature matrices in CI and actual-bundle reference checks; serialization/linear-gain checks; paired A/B/C development reports including same-candidate before/after reranking |
| M7: Serving reliability | Complete bundle validation, bounded requests, health/metrics, JSON logs, CPU Docker | Actual API/offline fixture predictions checked; incompatible bundle stays unready; corrupt/missing artifact tests; Docker smoke check; no raw queries logged |
| M8: MVP evidence and profiling | Fixed load workload, baseline stage profile, reproducibility instructions, error analysis | Repeatable CPU demo; all 3 development tracks; quality/latency/memory report; approved absolute ceilings and proposed optimization hypothesis; MVP checklist reviewed; test judgments locked |
| M9: Required inference optimization | One profiling-justified intervention; preapproved variants, controls, budgets, and adoption criteria | Completed reference/variant repeated benchmarks; C quality/latency tradeoffs and B retrieval evidence; concurrency failures; adoption or rejection decision; baseline retained when variants fail |
| M10: Reliability and portfolio evidence | Controlled failure/previous-bundle restore; required minimal inspector; technical case study; single frozen final evaluation | Measured rollback/recovery; all required experiment/review artifacts complete; approved final execution manifest; immutable A/B/C test report with uncertainty and limitations; one reproducible displayed two-pipeline comparison checked against actual API/report outputs; human review of claims |

## Dependency and stop points

The default order is M0 through M10. M5 starts immediately after M4 verifies retrieval baselines, before learned ranking and production serving. M6 builds only the thin route needed for parity tests; M7 completes operational serving. Keep review units smaller than a milestone when necessary.

The MVP's capability checklist is CPU-capable and doesn't require M5 weights. If GPU access or budget blocks M5, the lead may explicitly authorize M6–M8 using pretrained features; record M5 as outstanding. Resume M5, rebuild matching indexes, and repeat affected M6–M8 evidence before portfolio freeze. Never silently skip training or relabel a smoke test as completion.

The original M5 proposals and charge records remain historical evidence. For the completed sanity-then-training run, the user explicitly excluded unresolved earlier allocations and authorized practical execution limits: 3,600 seconds new allocation, 2,700 active seconds and 300 seconds finalization reserve, with at least 1,000 seconds setup allowance as needed. Observed release request was 622.98 seconds; conservative UI-confirmed charge was 650.72 seconds. Earlier charges were retained, not reset or fabricated. The broader six-hour budget and further catalog re-encoding remain unapproved; no historical all-runs ceiling compliance claim is made.

M8 never opens test judgments. Finish M9 and the controlled-failure exercise before M10's freeze. Select models and retained variants using development evidence, then obtain explicit approval for one final frozen A/B/C evaluation. A failed test gate is reported without retuning. If work stops at MVP, report development-only MVP completion and preserve test isolation.

## Independently verifiable completion criteria

| Evidence | MVP complete at M8 | Portfolio complete at M10 |
| --- | --- | --- |
| Data and methodology | Pinned corpus/splits, leakage checks, hand-verified metrics, reproducible manifests | Same contract plus one approved frozen final A/B/C test report |
| Retrieval and ranking | BM25, pretrained dense, RRF, XGBoost; A/B/C development reports | Also substantive M5 supervised PyTorch run, updated weights, resumable persistent checkpoints, pretrained/fine-tuned comparisons and matched bundles |
| Serving and correctness | CPU API/Docker, bundle validation, MLflow records, CI, exact feature-parity evidence | All affected checks pass for final bundles; controlled fault and measured previous-bundle rollback |
| Systems evidence | Stage profile and reproducible latency/throughput/errors/memory baseline | Completed M9 intervention with controlled quality/performance comparisons and justified adoption or rejection |
| Review and learning | Setup instructions, limitations, review packets for completed milestones; lead acceptance | Technical case study, AI-development/review log, all M0–M10 packets and human-understanding checks; approved portfolio claims |
| Minimal model-comparison demo | Optional | Required: one local view, two pipelines, top-10 products, versions/timings/judgment status; one reproducible displayed comparison verified against actual outputs |

Neither completion status requires an experimental model or optimization to win. A substantive negative result can satisfy its evidence requirement; a failed or unexecuted substantive run cannot. Promotion and completion are separate decisions.

## Required engineering learning/review packet

For every milestone, produce a concise versioned packet, provisionally `docs/reviews/M<n>.md`, linked to configs, reports, and relevant changes. Include:

1. Architecture explanation: data flow, interfaces, state, and why the design meets this milestone's goal.
2. Design alternatives: at least 2 plausible approaches and the reasons for choosing or rejecting each.
3. Failure modes: data/model/serving risks, how they're detected, recovery behavior, and residual limitations.
4. Evidence: exact commands, environment/configuration hashes, tests, experiment results, uncertainty, failures, and links to artifacts. For a planning milestone, state that no experiments ran and identify assumptions to verify.
5. Interview questions: 3–5 technical questions with short answer outlines grounded in this milestone's implementation or proposal.
6. Human-understanding checks: specific concepts and code paths the lead should explain or inspect before approval; record unresolved questions without pretending they were verified.
7. AI-development/review log: requested scope, generated changes, human corrections, accepted decisions, and remaining approval points. Record reviewable outcomes, not private model reasoning.

Review packets are milestone exit artifacts, not optional end-of-project summaries. Agent-generated answer outlines don't establish human understanding; the lead confirms the review. For M0, the current [plan's conflict resolutions and D1–D10](PLAN.md) are the decision record to review before authorizing M1.

## Testing depth

Use Pytest unit tests for metric math, grouping, normalization, fusion, identity mapping, supervision masks, checkpoint restoration, and feature construction. Add integration tests for artifact round trips and a tiny retrieval-to-ranking API pipeline. Use synthetic, human-auditable fixtures in routine CI; full data and model downloads aren't implicit CI dependencies.

At M6, compare the actual offline and HTTP ranker-input matrices for the same fixture bundle: names/order, dtypes, identity alignment, missing-value masks, and exact values. Exercise normalization, missing text, candidate permutations, batch sizes, and serialization; deliberately introduce a fixture mismatch to confirm detection. Any mismatch fails CI. Record actual pretrained/fine-tuned CPU-bundle integration checks outside routine CI. See [feature parity requirements](EVALUATION.md#exact-offlineonline-feature-parity).

Run static checks, fixture tests, and dependency checks on PRs. Add CPU container smoke checks when Docker exists. GPU tests and full-corpus evaluations run explicitly with recorded environments and budgets. Benchmarks produce reports outside ordinary unit-test timing assertions; shared CI runners aren't performance gates.

## Required minimal portfolio demo and later work

The required minimal inspector in M10 compares two selected retrieval/ranking pipelines in one local view, using the API or saved measured reports. Show product ranks, actual method/bundle IDs, development judgments or unknown labels, observed request/stage timings, and separately identified benchmark aggregates. Use development examples only; mark missing results as unrun, never simulated. It adds no training, ranking, or serving logic. UI technology needs approval; exclude accounts, dashboard suites, cloud deployment, and elaborate frontend work. Verify displayed results and timing provenance against the underlying outputs. It remains optional for MVP completion.

- Compare HNSW/IVF against exact nearest neighbors only if lookup time warrants it; report ANN recall separately from relevance recall.
- Add a cross-encoder only after establishing a reranking quality/latency budget.
- Select batching, reranking depth, or precision for M9 only when profiling justifies it. Further caching or quantization work needs a new study scope.
- Add dashboards, shadow execution, and automated rollback after the local versioned-release exercise works.

Kubernetes, Kafka, distributed GPU infrastructure, and unrelated features remain out of scope unless measured evidence justifies a separately approved design. Update documentation with each accepted milestone and explain why the next experiment is justified.

M6 native fixture update: [42 Colab CPU cases passed](reviews/M6.colab-boundary.md); actual M4/M5 bundle parity and fitting remain gated.
