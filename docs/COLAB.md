> Current 2026-10-29 status: the full-catalog L4 run and Colab CPU FAISS/development evaluation completed. See [M4.cpu-run.md](reviews/M4.cpu-run.md). The notebook instructions below are historical execution details; no additional run is authorized.

> Pilot result: [actual L4 evidence and Mac return review](reviews/M4.colab-pilot.md). The successful v5 snapshot fixes missing ensurepip using --without-pip and native pip; earlier v4/v3 exports remain historical. Full catalog is unapproved.

> 2026-10-28 update: the lead approved native Colab Python 3.13.16 after live discovery. The CUDA overlay now targets Python 3.13 with the same package versions and PyPI-verified wheel hashes; the Mac reference stays on Python 3.12. Input v4 and the separate ignored approved manifest supersede v3 for this bounded pilot. Full catalog remains blocked.

# M4 Colab integration and operator guide

Status: integration prepared for review. GPU execution is still blocked. The user selected Colab for neural embedding and later training; the Mac remains the development, CPU FAISS, evaluation and serving environment. M5/training requires its own approval. Read the [review packet](reviews/M4.colab.md) and [pilot proposal](../configs/m4-colab-pilot.proposed.json) before transferring or running anything.

## Boundaries and interfaces

The thin [notebook](../notebooks/atlasrank_m4_colab.ipynb) calls repository modules. `colab_prepare` exports the exact 4,096 canonical CPU calibration identities/titles, saved CPU vectors, CPU tokenizer outputs/hashes, pinned MiniLM files and an explicit source whitelist. It reads no judgments and runs no model inference. `cuda_encoder.CudaEncoder` inherits the existing model loader and uses the same tokenizer, preprocessing, model modules, masked mean pooling and L2 normalization. `configs/dense.json` and `encoder.py` remain unchanged.

CUDA is a distinct recipe with float32, eager attention, right padding/truncation to 256, batch 32, seed 42, deterministic algorithms, disabled autocast/TF32 and synchronized timing. Its recipe binds native Torch/CUDA/cuDNN/driver/GPU details, package versions and adapter/source hashes. CPU and CUDA chunks never share a job. A changed recipe, source, GPU/runtime, ownership or time budget stops resume.

`colab_runner` checks explicit configuration/storage approval and complete source/transfer pins before any install, directory creation or GPU work. It runs setup and the worker in subprocesses under a shared 600-second persistent session ledger. `colab_pilot` calls `encode_chunks`, with a callback that verifies persistence after each committed chunk. `colab_import` restores returned pilot chunks, checks exact identities/titles, compares CPU/CUDA vectors, and reconstructs/loads native CPU FAISS on the Mac.

## Environment specification

[environment.json](../environments/colab/environment.json) and [hashed requirements](../environments/colab/requirements.txt) define a separate CUDA overlay. The Mac `uv.lock` remains unchanged; its Linux Torch selection is CPU-only, so **don't run the project's uv sync in Colab**.

The initial candidate requires Linux x86_64, Python 3.12 and the user-reported native CUDA-enabled Torch 2.11.0+cu130. The user's NVIDIA-SMI output reports an L4 with 23,034 MiB and driver 580.82.07. That is operator-supplied evidence; the user subsequently reported Torch 2.11.0+cu130; actual Python and `torch.version.cuda` discovery remain unverified. The CUDA recipe preserves that exact native build. The Mac CPU calibration used Torch 2.14.1, so the numerical comparison includes backend and software-version differences; it doesn't isolate a pure device effect. If discovery reports another Torch/Python version, stop and review a revised exact environment before executing the pilot. Never install a CPU Torch build or silently upgrade/downgrade native Torch.

The runner creates a venv with access to system packages and installs 38 explicitly pinned, hashed overlay packages with `--no-deps --only-binary=:all:`. Torch and NVIDIA runtime packages are excluded. It verifies the native Torch file/version stayed unchanged and validates overlay versions before loading the model. Existing optional Colab packages can still cause import conflicts; treat those as failures for review, rather than uninstalling unrelated software. Full Linux/CUDA compatibility remains unexecuted.

PyTorch documents asynchronous execution, synchronization, TF32 controls and allocated/reserved allocator counters. The adapter synchronizes transfers/model completion and records allocated, reserved, peak and free-device memory separately. [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html), [CUDA API](https://docs.pytorch.org/docs/stable/cuda.html).

## Exact pilot proposal

| Setting | Proposed value |
| --- | --- |
| Device | Exactly 1 NVIDIA L4, `cuda:0`; stop if allocation differs |
| Input | Exact previously calibrated 4,096-title canonical sample; no labels |
| Model | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; same 11 verified files |
| Batch/chunks | 32 / 1,024 rows, 4 complete chunks |
| Precision | Float32, eager, no autocast/TF32, deterministic algorithms, cuBLAS `:4096:8` |
| Warmup/measurement | 2 warmup batches; 1 measured complete pass; 2 single-query repeat checks afterward |
| Time | 600 seconds cumulative managed session wall/GPU allocation upper bound, including setup, warmup, failures and restarts |
| GPU memory | 4 GiB allocator cap; require cap + 1 GiB free at load; stop below 1 GiB free or above cap |
| Host memory | 4 GiB available at load; 8 GiB worker RSS ceiling; stop below 1 GiB available |
| Storage | 1 GiB persistent run output, 3 GiB ephemeral working files; retain failures |
| Project transfers | Up to 256 MiB upload and 256 MiB return; model/input/source export measured locally |
| Dependencies | Up to 512 MiB downloaded wheel payload; sampled cap and retained setup logs |
| Persistent destination | Suggested private `/content/drive/MyDrive/AtlasRank/m4/pilot-l4-v1`; **unapproved** |
| Ownership | Approved stable owner token; operator attests exactly 1 active runtime |

Warmup is excluded from measured encoding throughput and included in the total session charge. Record setup/discovery/download/install durations, model load, tokenization, host/device copies, synchronized model inference, persistent checkpoint time, per-batch durations, new rows and measured titles/s. Report warmup/model-load/measurement memory separately. A resumed attempt reports newly encoded rows only; don't divide all 4,096 by a skip-only duration. Full-process timing and actual GPU utilization/kernel-active time are distinct; the ledger conservatively charges managed GPU allocation wall time. Operator time before the managed session is explicitly unmeasured and must be recorded separately for project accounting.

CPU/CUDA identities, normalized texts, token IDs/masks/types and model file hashes must match exactly. Require finite `(4096, 384)` float32 vectors, unit norms within absolute 1e-5, component rtol=1e-4 / atol=1e-5, and cosine absolute error <=1e-4. These are proposed cross-device gates, requiring lead approval; they don't replace existing CPU checks or M6 exact feature parity.

Compare top-20/top-100 membership/order on 16 label-free title-derived probes. Report differences, exact ties and cutoff changes. Reject omitted candidates or inversions separated by more than twice the cosine error bound; near ties remain visible and never trigger score rounding or metric changes. Repeated single-query CUDA outputs must agree exactly in the same runtime. The Mac repeats native FAISS checks using a fixed CPU query vector to isolate product-vector differences. Agreement isn't retrieval relevance evidence.

Stop on missing approval/storage, different GPU or environment, unavailable CUDA, incompatible source/model/tokenizer/recipe, invalid vectors/tokens, numerical-gate failure, OOM, memory/disk/time limits, checksum failure, Drive quota/I/O errors, or duplicate active ownership. No automatic fallback, precision/batch change, tolerance widening, budget reset or history deletion.

## Persistent transport and interruptions

Chunks are sealed and validated on the ephemeral POSIX workspace using the existing engine. The callback uploads each completed chunk to a fresh UUID ZIP in the approved private folder, rereads the compressed bytes and file hashes, and writes an immutable receipt last. It doesn't assume Drive supports POSIX rename, chmod, advisory locks or fsync semantics. Incomplete uploads remain present and aren't resumable chunks. Receipt-backed corruption fails closed.

Job/control files and attempt records are also immutable receipt-backed objects. A fresh workspace restores these and validated chunks before calling the same chunk engine, which checks exact source rows, titles, intervals and recipes before skipping them. Persistence exceptions stop before another encoding chunk. A failure after a local seal but before its persistent receipt can require re-encoding that **unpersisted** chunk after runtime loss. No acknowledged persistent chunk should be re-encoded.

The managed-session ledger lives directly at the approved destination. Start/end/recovery files charge setup and all retries; unclosed recovery conservatively includes downtime and can exhaust the 10-minute cap. That needs review, rather than a reset. Hardware/runtime changes also require review; incompatible outputs aren't mixed. The token is a procedural single-writer safeguard, not a distributed lock. You must stop the old runtime before resuming.

Drive mount writes depend on the runtime and are subject to quotas. A successful reread isn't proof of cloud durability. Before full-run approval, verify the completed pilot receipts from a fresh runtime or independent Drive download, stop the old runtime first, and run the local return checks. [Colab FAQ](https://research.google.com/colaboratory/faq.html). Actual Drive durability and CUDA interruption remain unverified until execution.

## Human operation

1. Review the source diff, [pilot manifest](../configs/m4-colab-pilot.proposed.json), environment and resource limits. Supply actual Colab Python/Torch/CUDA discovery. Approve the exact private account/folder and exclusive owner token. No approval is recorded yet.
2. Local preparation uses `python -m atlasrank.retrieval.colab_prepare --output .local/colab/atlasrank-colab-input-v3`. That destination has already been prepared for review; preserve it. A repeat needs a fresh name. The package creates the ZIP, file manifest, preparation evidence and a bound unapproved pilot JSON. No upload occurs.
3. After explicit lead approval, create a separate ignored approved manifest from the bound proposal. Set only `execution_approved`, `storage_approved`, the exact `persistent_destination` and agreed `operator_owner_token`. Keep all reviewed source/transfer/resource pins. Never change the checked-in proposal to approved. Stop if the native Torch/Python differs from the reviewed candidate.
4. Open the notebook privately in Colab and choose the allocated L4. Upload the reviewed ZIP and approved manifest via Files, paste the independently reviewed ZIP SHA-256 and matching owner token, and inspect discovery. No Git repository or remote is needed. Mount the approved Drive account and ensure the exact parent exists. Upload/mount/discovery preceding managed setup aren't claimed as measured execution.
5. Run the guarded pilot cell once. Inspect stdout/logs and persistent receipts. It includes dependency setup inside the 10-minute cap. On error, preserve the entire folder; stop the runtime. Review a restart's remaining charge before resuming with the identical manifest and token. A fresh workspace is required; a new managed attempt doesn't reset the ledger.
6. Stop the runtime after execution. Verify/download the complete persistent folder, including incomplete uploads and all ledgers, into a new ignored `data/esci/imports/m4-colab-pilot-v1`. Keep transfer checksums from Drive and compare them locally. No credentials, judgments or unrelated Drive files should return.
7. Run the bounded local check with the prepared references:

```sh
.venv/bin/python -m atlasrank.retrieval.colab_import \
  --store data/esci/imports/m4-colab-pilot-v1 \
  --inputs .local/colab/atlasrank-colab-input-v3 \
  --output data/esci/infrastructure/m4-colab-return-v1
```

This command remains unrun without actual returned artifacts. It verifies receipt/file hashes, exact canonical products and CPU comparisons, rebuilds CPU `IndexFlatIP`, verifies save/load/native row parity and records top-k differences. It accepts only the 4,096-row pilot. Send its report, managed-session logs and fresh-runtime durability evidence for lead review. A passing pilot doesn't authorize full catalog or M5.

## Full-catalog proposal after pilot review

The separate [full-run proposal](../configs/m4-colab-full-run.proposed.json) remains **unapproved and intentionally non-runnable**. Preserve all 1,215,854 products, full canonical order, 1,188 fixed chunks, exact recipe/environment/source pins, persistent receipts and cumulative wall/GPU accounting across sessions. No judged subset or precision/corpus reduction is allowed.

The matrix alone is 1,867,551,744 bytes (1.739 GiB). Propose 4 GiB persistent storage for compressed chunks, control/evidence and partial uploads, subject to actual pilot sizes. Local imported chunks plus matrix and native FAISS require roughly 5.22 GiB before compressed-return copies, mapping and reports. The existing 8 GiB incremental/8 GiB RSS/30 GiB total-data/2 GiB headroom limits still apply. Canonical projection and full local memory must be reviewed before creation.

A GPU wall/runtime number would be invented before this pilot. Keep those manifest fields null. Use measured setup + N/new-row pipeline rate + persistence + restart allowance + transfer/local import/index/evaluation, with explicit headroom and a separately approved cumulative GPU cap. No GPU speed or Colab session duration is assumed. Approval also requires a persistent destination, measured recipe ID, source/catalog projection hash, whole-project cumulative ledger and a separately reviewed full-run supervisor; the pilot runner rejects the full manifest.

Return only verified chunks, canonical mappings and evidence into a new local import, verify contiguous full coverage against the pinned catalog, then call the existing `build_bundle`/`load_bundle` on the Mac. Don't import a foreign native FAISS binary. Local CPU query encoding is paired explicitly with the CUDA product recipe through `evaluate_development(..., product_recipe=imported_recipe)`. The evaluator validates shared model/tokenizer/preprocessing semantics and records both recipes, never disguising CPU queries as CUDA. Fixture coverage exists; the actual pairing still needs numerical and development evidence.

Run all 4,129 development queries with the frozen A/B definitions, existing metrics/bootstrap/slices, BM25/dense/RRF-60, 100 candidates per retriever, and exact M3 per-query reference verification. No labels leave the Mac. Preserve M3, CPU calibration and every failed attempt. Full-catalog import/evaluation orchestration remains a separately reviewed execution step; this integration doesn't launch it or claim those results.
