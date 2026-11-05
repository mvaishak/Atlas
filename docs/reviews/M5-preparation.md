# M5 supervised dual-encoder preparation review

Status: v0.1 implementation preparation complete. The GPU pilot remains unapproved and unrun. M4 pretrained dense and RRF results remain frozen baselines. No official test judgments were opened.

## Design and data flow

`scripts/prepare_m5_export.py` reads only the pinned prepared training manifest and canonical catalog titles. It builds a deterministic, hash-ranked order over usable query groups and selected pairs, and emits a small immutable input ZIP containing training E positives, same-query explicit I negatives, their canonical titles, pinned model files, reviewed source snapshot, and environment pins. It excludes development labels, official test judgments, unselected catalog titles, and raw data. The full accepted catalog and M3/M4 bundle identifiers remain in provenance for later matched comparisons.

The training objective is a shared-weight MiniLM dual encoder initialized from `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Each group uses at most two E products and four admissible I products; all products that have E anywhere in the train manifest are removed from negative eligibility. No S/C/unknown labels, random negatives, cross-query in-batch negatives, or development supervision enter the loss. The per-query loss is `logsumexp(valid positive and negative logits) - logsumexp(positive logits)` at temperature 0.05. The batch loss is the equal-weight mean of query-group losses.

The reviewed pilot schedule is one deterministic pass over 9,180 usable groups: 574 query-group batches at batch size 16, including a final partial batch. AdamW uses learning rate 2e-5 and weight decay 0.01, 5% warmup with linear decay, gradient clipping at 1.0, float32, no TF32, and no autocast. The training loop advances the sampler cursor only after the optimizer and scheduler step; checkpoint global step must equal consumed cursor.

`src/atlasrank/training/` owns sampling, masks/loss, model calls, optimization, receipts, recovery accounting, export validation, and runner checks. The notebook only checks the runtime, installs the pinned overlay without replacing Colab's native CUDA Torch, stages the reviewed files to a private Drive folder, and calls the package runner. A separate approved manifest must be created only after the lead approves the exact proposal and destination.

## Data audit and export

Pinned source hashes and sampled export details are recorded in M5 validation evidence (`docs/reviews/M5-preparation.validation.json`, private evidence). The prepared train manifest has 336,694 rows and 16,759 groups. After train-global E masking and the required positive/negative availability, 9,180 groups are eligible; 7,579 groups are excluded, and 8,142 explicit I rows are masked because their products are E elsewhere in training. The deterministic plan contains 17,600 positive pairs, 26,998 negative pairs, and 43,509 selected product identities. It records a SHA-256 over its canonical bytes and a separate hash over materialized group records.

The immutable candidate export `.local/colab/m5-training-input-v3.zip` measured 87,664,999 bytes, under the proposed 512 MiB upload budget. It contains the pinned training-only pairs and selected titles, the model snapshot, code/environment pins, and provenance. It does not contain development judgments or test judgments. The export is local and ignored; its SHA-256 is in the validation JSON and pilot proposal.

## Checkpoint and recovery contract

Each write-once checkpoint is checksummed and contains model, optimizer, scheduler, Python/NumPy/Torch CPU/all-CUDA RNG states, sampler cursor, global step, and full run provenance. Checkpoints are organized by allocation session so a recovered runtime cannot overwrite prior evidence. The latest receipted checkpoint is eligible for resume only when source/configuration provenance matches. Recovery requires the operator to confirm that the old Colab runtime stopped; orphaned allocation is conservatively charged and the active lock is preserved as a recovery receipt. If a cap or integrity check fails, the runner stops while leaving prior artifacts intact.

A synthetic active-dropout test compares uninterrupted training with a step-2 interruption and resume. The predeclared maximum absolute parameter and optimizer/scheduler state difference is 1e-7; cursor and step equality are exact. The bounded test passed. This verifies deterministic mechanics on a tiny model, not GPU performance or model quality.

## Baseline comparison plan after training approval

The pilot is a training-mechanics run only and does not evaluate relevance. A completed fine-tuned model must be re-embedded over the same pinned canonical 1,215,854-product catalog, with a matching validated bundle. On development data only, compare M3 BM25, frozen M4 pretrained dense, frozen M4 pretrained RRF-60, fine-tuned dense, and fine-tuned RRF-60. Preserve Evaluation v0.3 Tracks A/B, gains, metrics, known-judgment policy, query normalization, paired comparison procedure, and M4 artifacts exactly. The official test partition remains untouched. No model promotion follows automatically from training or from a development improvement.

## Pilot request (not approved for execution)

The exact unapproved proposal is [configs/m5-colab-pilot.proposed.json](../../configs/m5-colab-pilot.proposed.json), SHA-256 recorded in the validation JSON. It proposes one NVIDIA L4 (23,659,151,360 reported total VRAM), native Python 3.13.16 / Torch 2.11.0+cu130 / CUDA 13.0, with runtime discovery repeated before execution. Limits are 3,600 cumulative allocated seconds including setup/retries, 2,700 active-runner seconds, 574 optimizer steps, 10 GiB peak allocated VRAM, 8 GiB minimum free VRAM, 16 GiB available host RAM at launch, 12 GiB worker RSS, 8 GiB scratch, 3 GiB persistent Drive storage, and 512 MiB each for upload and return. Prior M5 allocation recorded is zero; M4 allocation records remain separate and unchanged.

The storage plan estimates six immutable optimizer checkpoints at about 2.0 GB, final model and return archive at about 200 MB, plus the 87.7 MB input and small receipts/logs, all under the 3 GiB cap. These are conservative planning estimates, not measurements on Colab. The proposal uses `/content/drive/MyDrive/AtlasRank/m5/pilot-l4-v1`; the private account, empty destination, and unique exclusive-owner token must be confirmed/provisioned at approval. A future six-hour overall budget remains provisional and requires separate evidence-based approval.

## Alternatives considered

1. **Cross-query in-batch negatives.** Rejected because explicit query-specific I judgments are the approved negative source and other groups' positives may be semantically admissible or mislabeled as negatives. Per-query candidate masks preserve the supervision contract.
2. **Treat each positive independently with a one-hot cross entropy.** Rejected because it would make other E positives compete as negatives. The summed-positive log-sum-exp numerator treats the set of E products as one positive event, as approved.
3. **Save model weights only and restart optimizer state.** Rejected because it changes the optimization trajectory and can replay consumed data. Full optimizer/scheduler/RNG/cursor state and receipts support faithful resume.
4. **Implement training logic in notebook cells.** Rejected because it duplicates behavior and weakens reviewability. The notebook stays a thin runtime adapter over the package.

## Failure modes and limits

- Export/source/model/tokenizer mismatch: SHA and pinned identity checks stop before training.
- Mis-admissible negatives or query leakage: train-only partition checks, global-E mask, deterministic plan, and fixtures detect the issue; the plan is frozen before transfer.
- Nonfinite loss/gradients or invalid update order: training loop raises before committing a checkpoint; last immutable receipt remains available.
- Dropout/RNG resume drift: bounded deterministic fixture fails if parameter or optimizer/scheduler state exceeds 1e-7 or cursor differs.
- Drive collision, concurrent runtime, or unconfirmed stale lock: exclusive token, owner receipt, active lock, and explicit recovery stop the run.
- Budget exhaustion or low RAM/VRAM/storage: runtime guard halts and preserves prior checkpoints. A hard Colab VM loss may leave an orphaned charge that is conservatively counted.
- The proposed six-checkpoint footprint and runtime are not measured on real training. Pilot throughput, peak memory, loss trajectory, and checkpoint duration remain unknown.
- Development retrieval benefit cannot be inferred from a pilot. Full-catalog re-encoding and development Track A/B evaluation are separate work after pilot review and approval.

## Validation evidence

Commands and outcomes are in M5 validation evidence (`docs/reviews/M5-preparation.validation.json`, private evidence). Ruff and strict mypy passed for the new package/scripts; 11 focused tests passed. Notebook JSON and all code-cell syntax were checked without execution. Export, plan, source, model, M3/M4, environment, and evaluation-contract hashes were checked against the current repository and export. No GPU training, model fine-tuning, evaluation, dataset download, test-judgment access, push, or publication occurred.

## Interview questions and answer outlines

1. **Why mask every training I product that is E in any train query?** The product is known relevant elsewhere in training, so calling it a negative would create a false-negative gradient; the global mask enforces the approved admissibility rule.
2. **Why use `logsumexp(positives)` in the numerator?** It treats multiple E products as a summed positive event and avoids making one E product a negative for another; the denominator still includes all valid candidates.
3. **What does the 1e-7 resume test establish?** It checks deterministic optimizer continuation with active dropout in a synthetic fixture, including parameters and optimizer/scheduler state; it does not establish cross-GPU bitwise identity or quality.
4. **Why persist the sampler cursor with the optimizer state?** It identifies exactly which batches were committed, so resume neither repeats an update nor skips an uncommitted one.
5. **What evidence would justify a fine-tuned model over M4?** A matched full-catalog index and development-only Track A/B comparison under the frozen contract, with paired uncertainty and preserved BM25/pretrained references; training loss alone is insufficient.

## Human-understanding checks

- Trace the E-global negative mask from `data/supervision.py` through the exported immutable plan and verify why S/C are absent.
- Explain the numerator and denominator in `contrastive_group_loss` for a query with two positives and four negatives.
- Inspect the step/cursor commit order and describe recovery after interruption immediately before versus immediately after a checkpoint.
- Explain which bytes are included in the 3 GiB persistent budget and which figures remain estimates.
- Review the proposed L4 manifest, private Drive destination, cumulative budget accounting, and approval gate before training.

## AI-development and review log

- Requested scope: implement M5 training preparation only; no GPU training.
- Generated: package sampler/loss/model/loop/checkpoint/budget/export/runner modules, focused tests, local export script, guarded thin Colab notebook, unapproved manifest, and this review packet.
- Human design decisions preserved: explicit E/I supervision, summed-positive loss, float32 and fixed single-pass schedule, frozen M3/M4 references, exact evaluation contract, private/local workflow, and untouched test data.
- Remaining approval: approve one execution of the exact proposed manifest and private Drive destination, including the exclusive-owner token provisioning. No pilot artifact is approved by this packet.
