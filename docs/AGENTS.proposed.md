# Repository Guidelines

Approved D8 rules, amended in v0.3 for required minimal demo and provisional GPU budget. Root `AGENTS.md` is the operational guide; this file retains planning lineage.

## Purpose and ownership

AtlasRank must add defensible neural-modeling and systems evidence for Applied ML, ML Systems, Search/Ranking, and ML Platform roles. Prioritize correctness, reproducibility, and measured performance. The user owns architecture, evaluation methodology, and code approval; the agent implements approved work.

## Workflow

Read repository context and the approved plan before editing. Propose scope, interfaces, dependencies, tests, and acceptance criteria before substantial implementation. Work in small reviewable changes. Don't advance to another substantial milestone without authorization.

For every change, report files changed, commands run, outcomes, assumptions, and unresolved issues. Update relevant repository documentation. Explain important design choices for review and learning. Stop when the assigned scope is complete.

## Learning and completion

Every milestone requires a versioned engineering review packet: architecture/data flow, at least 2 design alternatives, failure modes, commands/tests/experiment evidence, 3–5 interview questions with answer outlines, and specific points requiring human understanding. Log AI-generated changes, human corrections, decisions, and unresolved questions. Mark unrun experiments explicitly. The lead confirms understanding; generated explanations don't establish it. Follow the packet requirements in [ROADMAP.md](ROADMAP.md).

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

Profile first, approve one bounded optimization study, then compare reference/variants under identical workloads and hardware. Report quality/latency tradeoffs and rejected variants. Freeze all final configurations before the single approved test evaluation; never tune on its outcomes or run an interim MVP test evaluation. Follow [EVALUATION.md](EVALUATION.md).

## Approvals and review

Obtain explicit approval for architecture or evaluation changes, major dependency replacement, GPU/cloud jobs, dataset/artifact/history deletion, pushes, deployment, and external-resource changes. Routine local edits and checks within an approved milestone may proceed.

PRs must state the problem, changes, validation evidence, experiment effects, and limitations. Link relevant issues and decisions. Use concise imperative commit subjects. Human review is required before code approval or merge; don't self-approve or promote models.

M2 must validate pinned XGBoost linear-gain NDCG with fixed synthetic predictions and establish query-specific negative-admissibility fixtures. M5 validates the actual sampler/loss locally before GPU execution. M1 does not implement metrics, samplers, data pipelines, or models.

## M1 foundation commands

Use the existing Conda Python 3.12 interpreter with project-local `.venv` dependencies. Follow the [README setup and check commands](../README.md). Pytest, Ruff, strict mypy, and Mac/Linux CI are configured. A reviewed `uv.lock`, actual MLflow round trip, and full checks are required before M1 acceptance; don't substitute direct version pins or source-only checks for those results.
