> Current 2026-10-29 status: full-catalog encoding used the approved L4 recipe; Colab CPU rebuilt FAISS and completed development Track A/B evaluation. See [M4.cpu-run.md](reviews/M4.cpu-run.md). MPS remains unrun and M5 is unapproved; the comparison proposals below are not authorization to execute.

# Optional M4 accelerator comparison plan

Status: historical optional comparison proposal, superseded for backend choice by the user’s decision to use Colab for neural workloads. [COLAB.md](COLAB.md) defines the prepared integration and unapproved L4 pilot. No MPS inference, Colab upload/execution, cloud spending, or accelerator full run has occurred. The CPU calibration remains the only measured model throughput: 44.5591 titles/s on 4,096 titles, fixed 256-token padding, batch 32, float32 and 1 CPU thread. It extrapolates arithmetically to 7.58 hours for full encoding, before full-run checkpoint overhead, sustained thermals, indexing and evaluation.

## Same-input CPU versus Apple MPS experiment

Request separate approval for **one comparison capped at 10 minutes total wall time**, including model loads, warmups, checks and all repetitions. Use 256 hash-selected titles from the existing pilot snapshot, identical ordered identities and text hashes, the pinned MiniLM revision, the same CPU tokenizer outputs, batch size 32, fixed right padding/truncation to 256, float32, eager attention, seed 42 and inference/evaluation mode. No shorter inputs, batch sweep, mixed precision, quantization or quality tuning. Add 4 synthetic normalization/empty/long-text cases and at most 16 synthetic or title-derived query probes without relevance judgments.

Run CPU and MPS sequentially in separate bounded processes. Use 2 warmup batches and 3 measured passes per device, with the order CPU/MPS, MPS/CPU, CPU/MPS. Record load time separately. Check `torch.backends.mps.is_available()` and supported deterministic operations before allocation; fail rather than silently fall back to CPU. The installed CPU adapter remains unchanged until an accelerator variant is explicitly approved. PyTorch documents MPS as a distinct device/backend with its own support requirements. [MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html).

Time both tokenization-to-CPU-vector completion and model-only work. Synchronize MPS before/after timed regions so queued GPU work and device-to-host copies aren't omitted; report tokens/s, titles/s, batch timing and all 3 repeat results. Record software, power/thermal observations where available, thread settings, input lengths, hashes, memory and errors. Three repeats support descriptive measurements, not a production percentile/SLO claim. The API provides synchronization and separate tensor/driver memory counters. [PyTorch MPS API](https://docs.pytorch.org/docs/2.14/mps.html).

Proposed correctness gates, registered before the experiment:

- Token IDs, attention masks and identity ordering must match exactly. Model/tokenizer revisions and pooling/truncation policy must match. All vectors must be float32, finite, 384-dimensional, and unit norm within absolute 1e-5.
- Keep existing CPU repeat/save-load checks. For the optional cross-device comparison only, propose component rtol=1e-4 / atol=1e-5 and maximum cosine-score difference 1e-4 on the fixed probes. Record actual worst/median errors and investigate failures; don't automatically widen tolerances. These proposed cross-device thresholds require review and don't replace the existing CPU/NumPy tolerance or M6's exact feature-parity requirement.
- Compare exact FAISS search on each device's returned vectors using the same CPU FAISS backend. Require rank agreement when score gaps exceed the declared numerical bound; disclose near-tie changes and computational top-100 overlap. That overlap is agreement with the CPU reference, not ESCI known-exact relevance recall. No labels are inferred from device agreement.
- Resume/chunk provenance must include the backend and full recipe. Never mix CPU and MPS product chunks in one artifact. A candidate accelerator-produced catalog is a separate bundle, with explicitly recorded CPU query encoding if that combination is proposed. Its correctness and development quality still need verification before adoption; a small throughput result doesn't promote it or satisfy M9.

Proposed limits: at least 6 GiB system-available RAM at launch; retain the 8 GiB process RSS ceiling and stop if available system RAM falls below 2 GiB. For MPS, cap allocations at the smaller of **2 GiB or 25% of `recommended_max_memory()`**, using the supported per-process fraction plus sampled tensor/driver counters. Stop if driver allocation exceeds that cap, if operations are unsupported, or on memory pressure. Apple GPU allocations share system memory, so report RSS and GPU counters separately and avoid adding them as independent RAM/VRAM pools. Keep incremental experiment files below 128 MiB within the existing data cap. No result or speedup is assumed.

## Whether Colab is worth using

CPU currently offers a verified local path with no transfer/setup cost. The user has selected Colab; the older MPS comparison remains optional and unexecuted. Colab is worth considering only if a separately approved small pilot on the **actually allocated GPU** demonstrates a useful end-to-end benefit after setup, transfer, checkpoint, restart and return costs. Google states that GPU types, usage limits and runtime lifetimes vary; no particular GPU, speed or session duration is assumed. [Colab resource FAQ](https://research.google.com/colaboratory/faq.html).

No Colab execution is included here. A future proposal should request explicit GPU/data-transfer authorization, identify the account and persistent checkpoint destination, and cap the first allocated-GPU pilot at 10 minutes including setup/warmup/retries. Start with the same bounded inputs and numerical gates, float32 with autocast/TF32 disabled, and a separately locked CUDA environment; the current Linux lock intentionally selects CPU Torch. Record the GPU model/VRAM and observed free space, reserving no more than the smaller of 4 GiB or half of available VRAM for that pilot. A full run needs a measured runtime/storage/billing budget and separate approval, with all M4 GPU use visible alongside the project GPU ledger. M5's training pilot and provisional cumulative budget remain unchanged and unauthorized.

After approval, transfer only the reviewed code snapshot/wheel, locked environment description, canonical title/identity file and pinned model files or fetch their exact public revisions. Transfer no raw examples, relevance judgments, credentials or Git remote. Save each completed immutable chunk plus manifests and attempt/recovery ledger to the approved persistent destination; never rely solely on Colab's temporary filesystem.

Return chunks, identities, model/tokenizer/recipe hashes, software/hardware metadata, timings and exported run records into a **new** ignored local import directory. Verify transfer SHA-256, inventory, complete row coverage/order, no duplicates/gaps, unit vectors and backend compatibility. Rebuild/load the native FAISS index locally rather than trust a foreign binary across incompatible versions, verify bounded cosine fixtures, and run local development-only A/B comparisons. Preserve the CPU reference and all failed/partial experiments. A backend or corpus mismatch fails rather than overwrite old artifacts.

Use measured rates and transfer sizes to estimate:

```text
total wall time = setup + input/model transfer + N / measured_titles_per_second
                  + persistent checkpoint I/O + output return + local validation/index/evaluation
```

A 384-dimensional float32 catalog matrix alone is 1.7393 GiB; chunks plus additional retained representations increase transfer/storage. The decision must include those costs and interruption risk. No cloud connection, upload, notebook creation or job was performed while preparing this plan.
