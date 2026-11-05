> M8 evidence, 2026-11-05. The approved CPU reference completed: 27 trials, 27,000 measured requests, exact instrumented HTTP/offline parity, 5.42 GiB sampled tree RSS and 20.54 MiB final persistent files. All 9,000 concurrency-1 attempts succeeded; concurrent stress retained 16,787 expected rejections. Shutdown and independent return checks passed; the owned pod was deleted. M8 lead acceptance and the proposed M9 study/absolute ceilings remain unapproved. See `docs/reviews/M8.md`.

> M8 execution authorization, 2026-11-05. The user accepted M7 and approved the M8 CPU reference benchmark and saved development analysis: 4 CPUs, 32 GiB allocated RAM, zero GPUs, 4 cumulative hours including setup/retries, at most 1 GiB new storage and at least 2 GiB estimated quota headroom. Preserve the 8 GiB available-memory and 12 GiB RSS gates. Reuse the existing M7 environment and immutable artifacts. M9 variants, model promotion, human code approval, milestone acceptance and the final frozen test evaluation remain separate gates. See `docs/M8_PLAN.md`.

> Latest authorization, 2026-11-04. The user authorized repository cleanup, correction of issues, and publication of the cleaned source to `https://github.com/mvaishak/Atlas.git`. This authorizes the required source commit and push to that origin. Data, model/index artifacts, generated evidence and logs remain private. GitHub Actions remains manual-only. M8, new model jobs, model promotion, human code approval and lead acceptance remain separate gates.
>
> Current evidence. M5 supervised training and M6 returned-model/saved-ranking checks are verified. The M7 real-model HTTP gate passed on DSMLP with exact offline/HTTP features and predictions. Docker fixture checks passed separately. See [the M7 review](docs/reviews/M7.md). Preserve test isolation, the native Mac prohibition and all historical evidence.
>
> Historical instructions below retain their original wording. The latest authorization above supersedes their source commit, remote and publication restrictions only for the cleaned source and user-designated origin.

> 2026-11-04 current handoff: the user authorized review/correction of `docs/M7_PROPOSED_PLAN.md` and execution of the next phase with self-review checkpoints. Supplied final M6 bytes and saved-ranking statistics are verified; see `docs/reviews/M6.return.md`. M7 is implemented for local review; Docker fixture checks and sealed-model native loading pass; actual full neural startup/parity remains unverified. Preserve the historical restrictions/evidence below, test isolation, native Mac prohibition and private/local scope. No new fit, model promotion, commit, publication or cloud allocation occurred.

# Repository Guidelines

D8 rules adopted following technical-lead approval. M3 remains committed locally as 2409da6. M4 full-catalog pretrained CUDA encoding and Linux FAISS/development Track A/B comparisons remain the accepted frozen baseline; see [M4 CPU run review](docs/reviews/M4.cpu-run.md). Under the user's explicit execution authorization, M5 completed a ten-update L4 sanity check and continued the same microbatch-8/accumulation-2 optimizer/scheduler state to 574 updates over 9,180 training groups. Returned checkpoint/model evidence is independently verified; see [M5 training review](docs/reviews/M5.training.md). The user then completed trained catalog encoding and development evaluation manually. M5.5 consolidates the available sources and evidence; see [M5.5 review](docs/reviews/M5.5.md) and [reproduction guide](docs/REPRODUCING_M3_M5.md). Local model pins, compact statistical replay and saved BM25 reference rankings are verified. Full historical M5 chunks/index/rankings remain externally stored and were not independently read in M5.5. Historical padding/normalization and FAISS tie-handling differences preclude a strict weights-only M4/M5 claim. The supported small signal is hybrid development Recall@100, not significant standalone dense improvement. The user subsequently approved the bounded M6 live-BM25/saved-development demo slice in docs/reviews/M6.plan.md. The demo is approved for local commit; see docs/reviews/M6.demo.md. The user approved the first shared-feature/exact-prediction-boundary phase; it is implemented for review in docs/reviews/M6.features.md. A zero-update XGBoost prediction attempt stopped on a preserved libomp failure; no retry/dependency change. The authorized deterministic fixture alternative is used. The shared-feature phase is committed as c13fe33 and the Linux runner as 11a607f. The synthetic zero-update native XGBoost boundary subsequently passed 42 cases on Colab CPU with independently verified returned evidence; see docs/reviews/M6.colab-boundary.md. Actual M4/M5 encoder-bundle parity and learned fitting remain separate gates; this Linux result does not resolve the Mac runtime conflict. Do not fit a ranker, retry native prediction, run full-catalog/model jobs, commit/push or promote models without further approval. Preserve the original failed run (1,944.79 seconds), setup stop (423.22 seconds), unresolved intervening allocations, successful-run charge and all prior history. The user explicitly excluded older allocations from the successful run's standalone one-hour cap; historical usage was not erased. The Mac Torch/FAISS OpenMP conflict remains unresolved: do not retry that combined execution path or modify its dependencies. Bounded saved-model CPU checks do not establish FAISS compatibility. Keep all artifacts local/private: no remote, push, publication or GitHub Actions. Full Linux CI, actual CUDA resume parity and MPS remain unverified. Preserve EVALUATION.md v0.3, the M3 reference, all M4 bundles/calibrations and every earlier experiment; never access official test judgments.

## Purpose and ownership

AtlasRank must add defensible neural-modeling and systems evidence for Applied ML, ML Systems, Search/Ranking, and ML Platform roles. Prioritize correctness, reproducibility, and measured performance. The user owns architecture, evaluation methodology, and code approval; the agent implements approved work.

## Workflow

Read repository context and the approved plan before editing. Propose scope, interfaces, dependencies, tests, and acceptance criteria before substantial implementation. Work in small reviewable changes. Don't advance to another substantial milestone without authorization.

For every change, report files changed, commands run, outcomes, assumptions, and unresolved issues. Update relevant repository documentation. Explain important design choices for review and learning. Stop when the assigned scope is complete.

## Learning and completion

Every milestone requires a versioned engineering review packet: architecture/data flow, at least 2 design alternatives, failure modes, commands/tests/experiment evidence, 3–5 interview questions with answer outlines, and specific points requiring human understanding. Log AI-generated changes, human corrections, decisions, and unresolved questions. Mark unrun experiments explicitly. The lead confirms understanding; generated explanations don't establish it. Follow the packet requirements in [ROADMAP.md](docs/ROADMAP.md).

MVP completion uses development evidence and keeps test judgments locked. Portfolio completion also requires substantive supervised PyTorch dual-encoder fine-tuning on bounded Colab GPUs after verified retrieval baselines, a profiling-led controlled inference optimization study, rollback evidence, a required minimal model-comparison demo, and one approved frozen final A/B/C evaluation. Smoke tests don't satisfy substantive training. Measured negative experiments may qualify; unexecuted work doesn't. Preserve the pretrained baseline and distinguish completion from model promotion.

## Code and configuration

Use typed Python, four-space indentation, `snake_case` modules/functions, and `PascalCase` classes. Keep data, retrieval, features, training, evaluation, artifacts, and serving separate under `src/atlasrank/`. Share offline/online feature code.

Use structured logging, bounded metric labels, validated configuration, and explicit error handling. Explain dependencies before adding them. Pin environments and version artifact schemas. Never commit credentials, private files, raw query logs, datasets, or generated model/index artifacts.

Keep one modular serving process. The required minimal portfolio comparison demo displays actual results and measured behavior through the API or saved reports; one local view, two selected pipelines, top-10 products, versions, timings, and development judgment status; no test judgments, invented values, accounts, dashboard suite, or duplicated ML logic. Optional for MVP only. Kubernetes, Kafka, distributed GPU infrastructure, and unrelated features require measured justification and separate approval.

## Testing

Use Pytest with `test_*.py` files. Add tests for relevant behavior, including failure paths. Prioritize metric correctness, query isolation, feature parity, artifact compatibility, and serving contracts.

CI must compare actual offline-evaluation and HTTP-serving feature matrices immediately before ranker prediction on the same pinned CPU fixture bundle. Require exact feature values, ordering, dtypes, missing masks, and product alignment; any mismatch fails CI. Exercise normalization, missing text, candidate/batch ordering, save/load, and mismatch detection. Approximate prediction tolerances cannot waive exact feature parity. Record actual-model CPU integration checks separately.

Use Ruff and mypy after approved setup. Keep routine CI CPU-based and fixture-only. Run applicable checks and document failures; never claim unrun checks passed. Full evaluations, downloads, and training require explicit execution scope. No coverage percentage is mandated initially.

## Experiments

Follow the approved evaluation contract. Never tune on test data, invent results, silently alter labels/splits/metrics, or label unjudged products irrelevant. Compare against frozen baselines and retain negative results.

Record data hashes, splits, seeds, model revisions, configurations, dependencies, hardware, checkpoints, and per-query metrics. Separate A labeled-candidate ranking, B catalog retrieval, C end-to-end retrieval-to-reranking, replay, and online experiments. Preserve known-exact Recall@20/@100 naming, unknown-label caveats, and paired uncertainty.

Approve training samples, schedules, seeds, runtime/storage caps, and persistent resumable checkpoints before Colab runs. Keep the pilot within 1 GPU-hour. Treat the 6-hour total as provisional pending pilot evidence and separate approval; track cumulative GPU use including restarts and catalog re-encoding. Rebuild matching embeddings/indexes/rankers when an encoder changes. Keep training logic inspectable in PyTorch; preserve failed runs.

Profile first, approve one bounded optimization study, then compare reference/variants under identical workloads and hardware. Report quality/latency tradeoffs and rejected variants. Freeze all final configurations before the single approved test evaluation; never tune on its outcomes or run an interim MVP test evaluation. Follow [EVALUATION.md](docs/EVALUATION.md).

## Approvals and review

Obtain explicit approval for architecture or evaluation changes, major dependency replacement, GPU/cloud jobs, dataset/artifact/history deletion, pushes, deployment, and external-resource changes. Routine local edits and checks within an approved milestone may proceed.

PRs must state the problem, changes, validation evidence, experiment effects, and limitations. Link relevant issues and decisions. Use concise imperative commit subjects. Human review is required before code approval or merge; don't self-approve or promote models.

M2 must validate pinned XGBoost linear-gain NDCG with fixed synthetic predictions and establish query-specific negative-admissibility fixtures. M5 validates the actual sampler/loss locally before GPU execution. M1 does not implement metrics, samplers, data pipelines, or models.

## M1 foundation commands

Use the existing Conda Python 3.12 interpreter with project-local `.venv` dependencies. Follow the [README setup and check commands](README.md). Pytest, Ruff, strict mypy, and Mac/Linux CI are configured. A reviewed `uv.lock`, actual MLflow round trip, and full checks are required before M1 acceptance; don't substitute direct version pins or source-only checks for those results.
