> 2026-10-29 status: full-catalog GPU encoding and Colab CPU FAISS/development evaluation completed. See [M4.cpu-run.md](reviews/M4.cpu-run.md). Do not retry the Mac command: native OpenMP failed and the user abandoned that path. Instructions below are retained as historical execution provenance.

# M4 full-catalog execution guide v0.1

Keep the repository and artifacts private. No training, M5, test judgments, remote or publication.

## Reviewed inputs and interfaces

The new input is `.local/colab/atlasrank-full-input-v4.zip`, containing exactly 1,215,854 canonical US product identities/titles, 11 pinned MiniLM files, reviewed Python source, CUDA environment specifications and provenance. Product columns are `product_locale`, `product_id`, `product_title`; order is strictly `(product_locale, product_id)`. No queries, relevance labels, split manifests, CPU vectors or development judgments are exported. The Mac retains those references. All 4,096 hash-selected CPU pilot identities/titles were found unchanged in this projection; they are not a catalog prefix.

The complete hashes and measured byte sizes are in the readiness evidence (`docs/reviews/M4.full-readiness.validation.json`, private evidence) and proposal. The archive is 149,122,132 bytes, expanded payload 158,276,718 bytes. The earlier full-input-v1/v2/v3 drafts are preserved; v4 includes the allocation anchor and the reviewed process/preflight safeguards. Preserve the older v1–v5 pilot exports and every earlier data preparation/experiment. This is a new export, not a replacement. Its input `CATALOG.json` pins every chunk's range and content hash; `FULL_TRANSFER.json` pins every exported file. The loader rejects extra files, symlinks, different rows, source bytes or recipes.

Model: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Batch 32, fixed right-padding/max length 256, existing NFKC/casefold/whitespace preprocessing, eager attention, attention-mask mean pooling, L2 normalization, float32, seed 42, no TF32/autocast. CPU and CUDA encoder/configuration bytes are unchanged from the accepted pilot. Chunks contain 1,024 rows: 1,187 full chunks and a final 366-row chunk `[1215488, 1215854)`.

Use [the thin full notebook](../notebooks/atlasrank_m4_full_colab.ipynb). It calls the package's separate full supervisor; it contains no ML/checkpoint logic. The pilot notebook and immutable pilot history remain available. No dependency was added. The cloud overlay uses the reviewed Python 3.13 environment with native Torch `2.11.0+cu130`, CUDA 13.0 and NVIDIA L4. It creates a venv with `--without-pip --system-site-packages`, downloads only hashed pinned overlay wheels and leaves native CUDA Torch intact. Exact accepted encoder/software/driver/CUDA-library recipe matching is required; a superficially similar L4 runtime can still fail and require review.

## Approval decisions before allocating a GPU

1. Review the source diff, full proposal, notebook and readiness packet. Approve the exact export/hash and unchanged numerical policy: component `rtol=1e-4`, `atol=1e-5`; cosine `atol=1e-4`; report near ties and reject inversion/omission gaps above `2*cosine_atol`. No threshold widening or rounding.
2. Confirm an exact **new private** Drive directory and intended Google account. Proposed example only: `/content/drive/MyDrive/AtlasRank/m4/full-l4-v1`. No destination has been selected or created. Never use the pilot directory. Retain this same destination across resumes.
3. Confirm total earlier GPU allocation, including manual discovery, idle time, failed setup and downloads. Known managed charges total 96.667498707 seconds. The proposal reserves **3,600 seconds total prior allocation**; this is a conservative proposal, not a measured total. It leaves four hours for new encoding/setup/return. If it cannot cover actual earlier use, revise and review the manifest before execution.
4. Confirm the proposed five-hour cumulative allocation/wall cap, 4 GiB Drive checkpoints, 6 GiB Colab scratch, 4 GiB GPU allocation and 8 GiB process RSS; approve the separate two-hour Mac phase and 8 GiB incremental local-artifact cap if desired. Mac start requires 6 GiB available RAM; runtime floor is 1 GiB available. Global data cap remains 30 GiB with 2 GiB free disk headroom.
5. Confirm exclusive single-runtime operation and prompt runtime deletion after returning evidence. The owner token is procedural ownership, not a distributed lock. The prior runtime must be stopped before resume.

After those decisions, prepare a **separate ignored** `.local/colab/m4-colab-full-run.approved.json` by copying the proposal. Generate an exclusive token locally with `secrets.token_urlsafe(32)` and put it only in that private file. Set `execution_approved`, `runnable`, `storage_approved`, `prior_allocation_charge_approved`, exact `persistent_destination` and `operator_owner_token`. Set `local_execution_approved` only if the lead separately approves Mac import/index/development evaluation. Preserve all other pins/limits. Have the lead review that concrete copy before using it. The checked-in proposal must stay false and contain no token.

## Cloud operator procedure — only after approval

1. Record the actual GPU runtime connection UTC epoch before uploads/setup. A notebook's first-cell clock can undercount already-connected idle time; supply the real connection time. Keep the same timestamp on a same-runtime retry and record each new runtime separately. The supervisor charges this pre-session allocation and gaps without double-charging closed sessions. The first full allocation timestamp is immutable. All gaps since then, including disconnected time and post-command idle, are charged conservatively without double-counting closed sessions; an overnight restart can exhaust the cap. This intentionally favors a safe upper bound over measured GPU utilization. Unclosed sessions also charge elapsed downtime conservatively.
2. Open the full notebook privately, select L4, supply the approved manifest and timestamp, and run discovery. Stop on runtime mismatch. If a runtime aborts before a managed session ledger is created, preserve its connection/disconnect records and obtain an allocation-charge review before retrying; its GPU time must not disappear. Upload `/content/atlasrank-full-input-v4.zip` and `/content/m4-colab-full-run.approved.json`. The notebook hashes the entire ZIP before executing its reviewed standard-library extraction helper, rejects traversal/duplicate/oversized archives, extracts fresh, and verifies all source/input pins. Do not run `pip install atlasrank`, install the Mac lock or replace Torch.
3. Mount the explicitly confirmed private account. Ensure the approved parent exists and the first-run destination is empty. Never overwrite an existing owner/configuration. The same token/destination is reused after a verified interruption; there is no lease stealing or full-manifest amendment mechanism.
4. Run the supervisor cell once. Equivalent command, from the extracted source with `PYTHONPATH` set to its `src`:

```sh
python -m atlasrank.retrieval.colab_full_runner \
  --manifest /content/m4-colab-full-run.approved.json \
  --inputs /content/atlasrank-full-input-v4 \
  --persistent /content/drive/MyDrive/AtlasRank/m4/full-l4-v1 \
  --work /content/atlasrank-full-work \
  --allocation-started-utc ACTUAL_CONNECTION_EPOCH
```

The destination above is only an example and must equal the approved manifest. The required environment variable `ATLAS_COLAB_EXCLUSIVE_RUNTIME` contains its private token; the notebook sets it. The supervisor includes discovery, dependency bootstrap, model load/warmup, chunk processing, checkpoint publication, validation and return packaging in the remaining budget. An independent parent/deadline watchdog stops the process group if its supervisor dies, including during native operations. Mac preflight failures also consume its separate ledger, and large hashes run inside the watched worker. No full workload can run from the proposal. Do not call the worker directly.

5. Watch JSON progress records. Batch component times distinguish tokenization, host/device transfers, synchronized inference and end-to-end encoding. Other records distinguish batch processing, resource checks, local chunk sealing and Drive publication/checksum overhead. Each published chunk gets an immutable progress summary. Timers are nested, not additive; stage totals and session wall time both matter. Supervisor logs/ledgers preserve failed attempts. A killed worker can lose its latest per-batch log tail, while completed chunks/receipts/progress remain.
6. On success, download `full-return.zip`, its `.transfer.json`, and the matching final supervisor `<attempt>.end.json`. The end record is written **after** return ZIP creation, so it is a separate required sidecar. All earlier root ledgers, acknowledged chunks and operational evidence are in the ZIP. Save the approved manifest too. Browser transfers are manual and may be slow. Record the actual allocation/disconnect time, account for post-supervisor idle/download time, and disconnect/delete the runtime promptly. The supervisor cannot end Colab billing after Python exits. Stop the runtime before five cumulative hours even if transfers are incomplete; Drive receipts remain available. Do not silently exclude idle time from the final budget.

On failure, preserve every attempt, ZIP, partial upload, receipt and ledger. Stop the old runtime. A fresh runtime restores acknowledged chunks and original job/control records, validates their exact source/title/recipe alignment, and skips them. Upload without receipt is preserved but cannot be skipped. A chunk sealed only on ephemeral disk may need encoding again if that VM is lost. Changed runtime recipe, checksum failure, exhausted conservative budget, insufficient storage, concurrent ownership or an ambiguous clock requires review. Never reset counters, delete history, change batch/precision, or reduce the corpus to continue.

## Return and Mac procedure — separate approval required

The return archive contains receipt-backed chunks/control/evidence and root session records; unacknowledged uploads stay on Drive and are listed. Its external checksum must be checked independently on the Mac. Preserve the untouched downloaded archive, transfer JSON and final session end JSON under a fresh ignored `data/esci/imports/` location. Review the cumulative allocation evidence, including manual connection/disconnect records. Do not claim actual hours from managed seconds alone.

Only after local execution approval and at least 6 GiB available RAM:

```sh
.venv/bin/python -m atlasrank.retrieval.colab_full_local \
  --manifest .local/colab/m4-colab-full-run.approved.json \
  --inputs .local/colab/atlasrank-full-input-v4 \
  --archive data/esci/imports/m4-full-l4-v1/full-return.zip \
  --return-sha256 INDEPENDENTLY_VERIFIED_RETURN_SHA256 \
  --work data/esci/infrastructure/m4-full-local-v1
```

The local supervisor charges import, validation, index build, model load, development evaluation and retries to a separate two-hour ledger. It streams nested ZIPs to avoid keeping a second compressed catalog copy. It checks the external/member/receipt hashes, validates complete chunk ownership/ranges/identities/titles against every canonical row, checks the saved hash-selected CPU pilot with the original thresholds, builds native CPU `IndexFlatIP`, verifies every reconstructed vector against the matrix, and runs the existing development-only A/B evaluator. Product CUDA and query CPU recipes remain explicit in provenance. BM25 scores/ranks/metrics must exactly match the frozen M3 per-query reference. RRF uses the frozen settings; no test or Track C quality evaluation runs.

The downloaded archive, restored job and native bundle are retained. Partial local attempts are retained; a retry uses a new attempt directory under the same cumulative ledger. It does not currently resume a partially built native FAISS bundle or partial evaluation. Repeated partial attempts can exhaust the disk reservation; stop for review rather than deleting them. A later source update requires a new reviewed export/manifest, never an invisible patch to this archive.

## Stop and review

Stop on failed approval/source/runtime/numerical/receipt/coverage checks, OOM, unavailable RAM, disk headroom/data cap, Drive quota/error, time cap or duplicate ownership. A budget cap can leave an incomplete but resumable job. Full-scale throughput, peak Mac memory, Drive overhead and completion within the proposed caps remain unverified. Do not start an optimization sweep, fallback backend, training or M5.
