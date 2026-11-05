> Current 2026-10-29 status: full-catalog GPU encoding and Colab CPU FAISS/development A/B evaluation completed. See [M4.cpu-run.md](reviews/M4.cpu-run.md). The plan below preserves the earlier infrastructure proposal and is historical; M4 awaits technical-lead review, M5 remains unapproved.

# M4 dense and hybrid baseline plan

Infrastructure implementation scope, approved after v0.2 calibration review: add immutable checksummed embedding chunks and a cumulative run ledger; verify interrupted/resumed parity with fixed batch/chunk boundaries; validate a dense bundle's lineage, identities, vectors and native FAISS rows; implement guarded development-only A/B comparison without changing metrics. Use synthetic fixtures and saved pilot vectors for bounded integration. No additional dependencies, full-catalog encoding, accelerator execution or M5 work are included. Prepare an optional same-input CPU/MPS/Colab comparison plan; execution remains gated.

Historical plan status: v0.3 described the infrastructure stage before full execution approval. Immutable chunks, dense bundle validation, guarded A/B evaluation and the source-pinned CPU supervisor were later used for the completed run linked above. Do not use this historical resource proposal to launch another full run.

## Model and provenance

Use `sentence-transformers/all-MiniLM-L6-v2`, revision **`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`**. The [revision-specific model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md) identifies Apache-2.0 licensing, English sentence embeddings, a MiniLM-L6-H384-uncased initialization, and contrastive training on over 1 billion sentence pairs. This is externally pretrained provenance; AtlasRank won't reproduce that training or claim ESCI-specific supervision. Training-data overlap with ESCI isn't established.

Read-only Hub metadata and small configuration files were inspected. The pinned modules specify a 6-layer, 384-dimensional Transformer, attention-mask-aware mean pooling, and normalization. The Sentence Transformers maximum is **256 wordpieces**, despite the tokenizer/model capacity of 512; use 256 including special tokens. The tokenizer is uncased. Preserve the pinned card/configuration files and attribution alongside the local model snapshot. The selected revision has no standalone LICENSE/NOTICE entry in its file listing; record the card's license declaration and link to the [Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0). Publication and redistribution remain separately gated.

The single selected weights file is `model.safetensors`, 90,868,376 bytes, with upstream LFS SHA-256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`. Verify downloaded bytes against that hash. Download only this file and the explicitly listed tokenizer/configuration/card files at the pinned revision, roughly 92 MB total. Exclude alternate PyTorch pickle, ONNX, OpenVINO, TensorFlow, and Rust weights. Use `trust_remote_code=False`, safetensors-only loading, and local-only loading after acquisition. Never load the repository's training script.

## Interfaces, dependencies, and boundaries

Extend existing `retrieval/`, `artifacts/`, and evaluation-job boundaries. Add an encoder adapter, exact dense adapter, and pure RRF helper; keep one process and retain the M3 interface of stable locale/product IDs, scores, and titles. No new service, frontend, feature/ranker pipeline, or architecture expansion is proposed. M4 can stay offline; M6 retains its separate actual HTTP ranker-feature parity requirement.

The following pins were resolved and installed locally during the approved setup. Their initial selection used upstream package metadata; runtime evidence is recorded in `docs/reviews/M4.validation.json` (private evidence):

| Package | Selected pin | Purpose |
| --- | --- | --- |
| torch | 2.14.1 on Mac; matching 2.14.1+cpu Linux distribution | Inspectable frozen float32 inference |
| sentence-transformers | 6.1.0 | Load the pinned Transformer/pooling/normalization recipe |
| transformers | 5.20.2 | Pinned tokenizer/model implementation |
| huggingface-hub | 1.33.0 | Revision-specific local acquisition; satisfies both model libraries' version ranges |
| safetensors | 0.8.0 | Non-pickle model weights |
| faiss-cpu | 1.15.1 | Exact normalized inner-product index |

All 109 preexisting lock package versions were preserved; the new lock has 131 package entries, including platform-specific PyTorch variants, and the Mac environment passes `uv pip check`. Local Torch/FAISS/Sentence Transformers/Transformers imports and the quality checks passed. Linux uses the [official CPU PyTorch index](https://download.pytorch.org/whl/cpu/torch/), with no CUDA dependency entries in the lock. Latest Hub 2.x is incompatible with this Sentence Transformers requirement; the selected Hub pin is deliberate. Wheel presence doesn't establish Linux runtime or ABI validation; Linux runtime/CI remains unverified, with no GitHub Actions launch.

## Encoding and resumability

- Product text is title only from the exact approved `prepared-v4` catalog, in ascending `(product_locale, product_id)` order. Apply the existing NFKC/casefold/whitespace normalization to titles and queries, retain punctuation for the pinned WordPiece tokenizer, add no prompts or field concatenation, and record that recipe separately from BM25's token extraction.
- CPU reference: float32, eager attention, `eval()` and inference mode, seed 42, deterministic algorithms, 1 PyTorch/BLAS/OpenMP thread, tokenizer parallelism disabled. Product batch size 32; query batch size 1. Right truncation and fixed right padding to 256 tokens remove length-dependent padding changes. Record all settings and truncated/empty-text counts. Retain every product, including empty text; encode it using the same special-token policy, and reject any nonfinite or zero-norm result rather than silently drop a row.
- Stream titles and token batches. Write 384-dimensional float32 unit vectors into bounded immutable chunks plus a final memory-mapped matrix. Persist each chunk's row interval, ordered identity/text hashes, vector hash, encoder/tokenizer/config hashes, backend, and completion record. Resume only verified completed chunks with identical settings; preserve failed partial files. Tokenizer/model buffers and the whole catalog must never become a single tensor.
- Count each canonical row exactly once and bind the final vector matrix, row mapping, and FAISS contents to the same corpus/text/model hashes. Keep all M3 bundles and previous preparations/reports unchanged. New artifacts go under new ignored `data/esci/` destinations; source/configuration/provenance summaries alone are reviewable Git content.

CPU is the proposed default for both product and query encoding, avoiding an unmeasured cross-device reference. Native MPS is an optional separately approved local experiment: check support, memory and CPU agreement on the same bounded sample before using it, record its own backend/precision/bundle identity, and never mix product chunks from different backends. Do not silently fall back between devices. Colab is an alternative requiring explicit cloud/GPU and data-transfer approval; no upload, cloud execution, or spending is included here. M5's 1-GPU-hour training pilot and provisional 6-hour estimate remain separate.

## Resource plan and execution gates

The initial planning snapshot was Apple M4, 16 GiB RAM, about 3.08 GiB available RAM and 62.56 GiB free disk. Existing data/history then used 3,938,007,654 bytes (3.67 GiB). After the lead closed apps, available RAM exceeded 5 GiB and the pilot passed its 4 GiB start gate. Current observations and installed versions are recorded in validation evidence; these are point-in-time values rather than reservations.

For 1,215,854 products at 384 float32 dimensions, one matrix requires exactly **1,867,551,744 bytes (1.7393 GiB)**. A flat FAISS index adds approximately another 1.7393 GiB for vectors, per [FAISS storage documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes). A retained chunk copy can add another matrix's worth. Product mappings, model files, reports, checkpoint metadata, and temporary writes need additional space.

| Phase | Proposed bound or estimate |
| --- | --- |
| Dependency environment/cache growth | Reserve up to 4 GiB outside the data-root reservation; actual lock/download sizes checked first |
| Bounded CPU calibration | At most 4,096 hash-selected catalog titles, 10 minutes of encoding, 1 GiB incremental data reservation; no relevance-label selection |
| Full artifacts, including resumable chunks | Reserve 8 GiB additional within the existing 30 GiB data cap; projected total about 11.67 GiB, including old history |
| Encoding RSS | Estimate 2–4 GiB with bounded batches; measure before extrapolating |
| Index-build/evaluation RSS | Estimate 4–6 GiB including index, mapping, lexical state, and buffers; avoid simultaneous full-matrix copies |
| Existing hard stop policy | Keep the 8 GiB sampled process ceiling, 30 GiB data cap, and 2 GiB free-disk headroom; retain the proposed 6 GiB serving ceiling for future combined serving |

The approved calibration encoded 4,096 hash-selected titles in **91.9228 seconds (44.5591 titles/s)**. The whole monitored command took 99.7148 seconds, with sampled peak child RSS **1,101,430,784 bytes (1.03 GiB)**. Attention-mask pooling/normalization, CPU repeatability, and single-versus-batched synthetic comparisons passed. The sample's token counts including special tokens were p50=26, p95=50, maximum=104; no sampled title was truncated or empty. A synthetic long input separately exercised truncation. The controller enforces a stricter 10-minute whole-command cap, while the inner timer also caps encoding at 10 minutes. It preserves partial files on failure; this pilot command doesn't implement full-run chunk resume.

**Stop after calibration and present measured throughput, peak memory, token-length/truncation distribution, and extrapolated full-run cost for approval.** No full-catalog encoding is included in that calibration approval. For a full run, propose at least 6 GiB available system RAM at launch, incremental disk reservation, a reviewed wall-time cap, and periodic checkpoints. If the full snapshot doesn't fit, stop instead of changing the corpus or evaluation scope.

The measured rate extrapolates arithmetically to **7.58 hours for full-catalog encoding alone**. The inner timer includes fixed-shape inference, resource checks, vector writes, and flushes; setup, sampling, model load, and token-length analysis are outside it. The short sample doesn't establish sustained thermal throughput or full-catalog length/truncation distribution. No automatic device switch, shorter truncation, batch-size sweep, or quality tuning follows from this result.

Proposed next execution budget, pending review: one CPU baseline run with a **12-hour cumulative command wall-time cap across encoding, index build, and A/B evaluation, including interruptions/restarts**, the existing 8 GiB sampled RSS ceiling, **8 GiB additional artifact reservation** within the 30 GiB data cap, 2 GiB free disk headroom, and at least **6 GiB available system RAM at launch**. At 50% longer encoding time than this extrapolation, encoding would take about 11.37 hours; the remaining time isn't guaranteed to cover other work. Stop and preserve progress if the cap is reached. The chunk, bundle and evaluator components are now implemented with synthetic and saved-pilot replay verification, including a real child-process interruption. The complete full-catalog worker remains unexecuted. The source-pinned proposal in configs/m4-full-run.proposed.json fails closed while execution_approved is false. Record actual cumulative runtime rather than resetting the ledger on resume. No MPS/Colab or extra full encoding repetition is included.

## Exact retrieval and deterministic fusion

Build CPU `faiss.IndexFlatIP(384)` by adding canonical normalized vectors in bounded chunks. Unit-vector inner products implement cosine similarity; reject dimension, norm, dtype, identity, and finite-value mismatches. Preserve the row mapping explicitly rather than infer identity from file order. Validate index type, metric, dimension, `ntotal`, component hashes, and reconstructed vectors against persisted rows before readiness/use. Treat native index loading as trusted local input behind checksum validation.

Queries use the same pinned encoder/preprocessing and normalization, with no dropout, prompts, or online fitting. Return exactly min(k, catalog size) dense candidates, including negative cosine scores if present. BM25 retains its frozen positive-match-only behavior. For equal dense scores, ascending product key resolves ties, including ties at the retrieval cutoff. Plan a threshold search plus range retrieval including all threshold ties, followed by canonical sorting; verify the pinned FAISS search/range numerical behavior on duplicate-vector fixtures before accepting this implementation. Sorting only the library's initial k hits is insufficient. If threshold completeness or consistent scoring can't be established, stop for a reviewed deterministic exact alternative rather than silently discard ties. Fix CPU query batch size to 1; cross-platform near-tie changes remain a measured numerical limitation.

The implemented adapter passes synthetic cutoff-tie fixtures and three pilot-vector queries. On one query, NumPy separated a pair by 2.98e-8 while FAISS produced an exact float32 tie and sorted by canonical identity. Top-100 membership matched for all 3; the largest score discrepancy was 1.19e-7. The original overstrict exact cross-kernel order assertion remains recorded as failed. The follow-up uses the plan's unchanged rtol=1e-5 / atol=1e-6 score tolerance, requires agreement for well-separated scores, and still requires canonical actual ties, exact FAISS repetition and exact saved-vector/save-load parity. It doesn't alter ranks, round scores, change the evaluation contract, or relax M6's exact feature checks.

Track B uses the frozen 100-per-retriever budget, deduplicated union, ranks starting at 1, and `RRF(d) = sum(1 / (60 + rank(d)))`; absent candidates contribute 0. Keep the top 100, breaking exact fusion ties by product key. BM25 may return fewer than 100; never pad it. RRF scores aren't calibrated probabilities and don't combine raw BM25/cosine magnitudes. Freeze these settings without a sweep.

## Evaluation and tests

Use guarded, hash-verified development manifests only. In Track A, score every supplied judged product using full-catalog BM25 statistics and dense cosine. Construct each method's full judged-pool ranking before RRF over those same candidates; don't truncate A to B's retrieval budget. In Track B, retrieve from the full catalog without supplying judgments, then compare BM25, pretrained dense, and their RRF using frozen known-exact Recall@20/@100, coverage, returned counts, slices, and paired deltas/95% group-bootstrap intervals (2,000 resamples, seed 42). Preserve unknown labels and all denominators. Assert the BM25 reference reproduces the saved M3 semantic results. Report hybrid candidates as the future C pre-ranker reference; no learned-reranking or completed C claim.

Routine tests use synthetic fixture vectors/encoders without downloads. Separate explicitly executed integration checks use the pinned actual model. Proposed acceptance tests:

1. Hand cosine and RRF fixtures; FAISS scores match NumPy brute force with rtol=1e-5 / atol=1e-6, unit norms within atol=1e-5, and identical IDs/order for fixtures whose score gaps exceed numerical error. Exact ties require canonical identity order, including over 100 duplicate vectors at the boundary. Duplicate/missing/unknown identities fail.
2. Encode repeatability on the fixed CPU reference, Unicode/whitespace inputs, empty text, long truncation, negative similarities, query isolation, product input reordering, and alternate batch/request shapes. Start actual-model batch comparisons with rtol=1e-5 / atol=1e-6; failures require investigation, not a silent tolerance increase. Any later MPS comparison has a separately registered tolerance and never relaxes M6's exact feature-parity contract.
3. Saved vectors and reconstructed FAISS rows agree exactly with their stored float32 representation; fixed saved-query search repeats and save/load preserve IDs/order. Corrupt files, mismatched encoder/tokenizer/corpus/backend settings, incomplete chunks, wrong dimensions/norms, and invalid index types fail closed. Interrupted/resumed fixture encoding must reproduce an uninterrupted artifact.
4. Independent A/B evaluation paths obey their distinct candidate policies. Fusion handles absent lists, overlap, ties, empty BM25, and input permutation without injecting positives. Integration compares the actual model recipe to attention-mask mean pooling plus L2 normalization.
5. Existing Pytest/Ruff/formatting/strict mypy/dependency/build/MLflow checks pass. Record actual timings/resources, downloaded file hashes, package versions, query-level development outputs, negative results, and all unrun checks. No minimum relevance improvement is required; correctness and faithful measured comparisons establish M4 completion.

## Delivery and review

Finish with versioned source/configuration changes, immutable local dense/hybrid artifacts, `docs/reviews/M4.md`, aggregate evaluation evidence, and `M4.validation.json` linking actual commands/logs/hashes. The current M4 packet is planning-only. Include architecture/data flow, at least 2 alternatives, failure/recovery limits, 5 interview questions, and human-understanding checkpoints. Retain local per-query data and previous history; no remote, push, publication, GitHub Actions, fine-tuning, learned ranker, test access, or M5 work.

## v0.3 infrastructure and execution recommendation

See [DENSE.md](DENSE.md) for chunk publication, exact row alignment, cumulative restart accounting, bundle compatibility and the development-only evaluator. The bounded 64-row saved-vector replay was signal-terminated during chunk 2 and resumed; chunk hashes/mtimes and all 5 final bundle files matched the uninterrupted reference. No new model inference or real judgments were used by that check. Full-catalog sorting/index/memory performance, actual-model long-run resume behavior, and A/B quality remain unmeasured.

Recommend the existing measured CPU recipe for the first full reference run: 1,215,854 approved US products, batch 32, chunks of 1,024 rows, float32 normalized 384-dimensional vectors, CPU IndexFlatIP, RRF-60 with 100 candidates per retriever, and all 4,129 development queries only. Preserve the M3 reference and require exact per-query BM25 score/rank/metric parity against its pinned development report. Full-run authorization is still pending; create a separate ignored approved manifest only after lead review, keeping the reviewed source/configuration hashes and budget unchanged.

The requested budget remains 12 hours cumulative active command wall time across encoding/index/evaluation and restarts; 8 GiB new run-directory storage; 8 GiB sampled worker RSS; 6 GiB available RAM at launch; 1 GiB available-RAM stop threshold; existing 30 GiB total data cap and 2 GiB free disk headroom. Retained chunks plus the matrix and FAISS vectors alone use about 5.22 GiB; mappings/canonical input/reports/failed partials share the remaining reservation. A large failed bundle can exhaust that allowance, requiring review rather than deletion. The 7.58-hour encoding extrapolation excludes full-run checkpoint and sustained thermal overhead. The 12-hour cap is a stop condition, not a completion guarantee.

An optional 10-minute, 256-title same-input CPU/MPS comparison could inform a later backend choice, with explicit numerical and unified-memory checks. [ACCELERATORS.md](ACCELERATORS.md) also gives conditional Colab transfer/return and accounting requirements. Neither option is authorized or executed, and no GPU speed/session duration is assumed.

## Colab execution decision

The user selected Colab for neural inference and later training, with the Mac retaining development, native FAISS, evaluation and serving. See [COLAB.md](COLAB.md) and [the integration review](reviews/M4.colab.md). Preparing the integration is authorized; actual L4 pilot configuration/storage approval remains outstanding. Preserve the CPU calibration and recipe. The older CPU full-run proposal remains unapproved and is no longer source-compatible after integration changes. Full catalog and M5 remain gated.
