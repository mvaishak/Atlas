# M8 CPU profiling and development evidence

The user accepted M7 and approved M8 on 2026-11-05. The scope is a CPU reference benchmark, stage profiling and development error analysis. The execution budget is 4 CPUs, 32 GiB allocated RAM, zero GPUs, 4 cumulative hours including setup/retries and at most 1 GiB of new persistent files. Preserve 2 GiB of estimated quota headroom. Effective available-memory admission remains 8 GiB and sampled/process high-water RSS remains limited to 12 GiB.

## Workload and measurement

Hash-select 1,000 normalized development query groups with seed 42 from the hash-verified development file. Store source/query hashes and exclusion counts privately. Each pipeline uses the identical query sequence. The selected reference pipelines are BM25, M4 RRF and experimental M4/M6 XGBoost, with frozen retrieval-100, RRF-60 and reranking settings, returning top 10. Test judgments stay locked.

For each pipeline, measure concurrency 1, 4 and 8 with 3 repeats. Before every measured repeat, issue 100 sequential warmup requests, followed by 1,000 measured requests. There are 27 measured trials and 27,000 measured requests, plus 2,700 warmups. Rotate pipeline execution order by repeat. A separate client process sends closed-loop requests to a single uncached Uvicorn worker. Use the existing 8-connection limit and 30-second request timeout.

Record every attempted request's status, timing, version identity and query hash. Retain failures and distinguish application busy responses from Uvicorn connection admission responses. Report all-request, success-only and error latency distributions separately, successful requests per second and each repeat's counts. Fast rejection latency cannot establish successful inference latency. These shared-node observations don't establish reserved-core performance.

## Profiling boundaries

Production timing fields add feature construction, ranker prediction, ranking order and request dispatch. Existing aggregate timings remain available. Request dispatch includes middleware, validation and threadpool dispatch. The benchmark wrapper measures query encoding calls without changing encoder source, recipe, inputs or vectors. Feature construction includes a second query encoding in the learned pipeline. Encoding totals and aggregates overlap, so they must not be summed into wall time. Client/network/serialization residuals remain separate from directly observed server dispatch.

Before load, the real instrumented model must pass exact actual-HTTP/offline feature and prediction parity. Record cold startup/readiness separately. A supervisor samples the server/client process tree every 0.1 seconds, applies the fixed deadline and disk caps, verifies model hashes and records shutdown. Setup, partial failures and repeats remain preserved. Reuse the existing M7 environment and artifact tree.

## Quality and storage

Use previously verified development A/B/C results, with explicit saved-result provenance. Recheck M6 per-query hashes and normalized group identities, then report errors by normalized whitespace query length and largest saved C metric differences. Preserve the original M4 C status and all incomplete-judgment caveats. This analysis creates no new retrieval rankings or relevance claims.

Actual per-user storage quota isn't exposed by filesystem free-space reporting. Use the smaller of filesystem space and a conservative 4 GiB baseline derived from the user's approximately 5 GB remaining, subtracting every new M8 file including failed outputs. The 1 GiB output cap retains at least 2 GiB under that conservative baseline. Capture this limitation with the run metadata.

The source proposal remains unapproved. The private execution copy records the user's authorization, deadline, exact source/workload hashes and admitted pod metadata. The recorded launcher uses CPU request 2 / limit 4 and memory request 16 GiB / limit 32 GiB. Effective cgroup limits are verified before model loading. Existing private artifacts and historical results remain separate from generated M8 evidence.

## Alternatives and acceptance

Directly timing model helpers would omit HTTP admission and serialization. The separate HTTP client preserves those boundaries. A full native profiler on every measured request would change the workload overhead, so the reference uses lightweight stage timers and clearly reports their boundaries. Increasing request concurrency or changing admission would create an optimization variant. M8 retains the current serving policy and reports its failures.

Acceptance requires complete repeat-level results, source/workload/environment hashes, preserved parity, resource and shutdown evidence, development quality/error analysis, and a concrete profiling-led M9 proposal. Absolute performance limits and M9 variants require review after these baseline observations. M8 does not authorize M9 execution, promotion or final test evaluation.

## Reproduction procedure

The checked-in `configs/m8-benchmark.proposed.json` deliberately has `approved: false`. Copy it privately only after a concrete run authorization, set the absolute cumulative deadline once, register the workload and source-manifest SHA-256 values, and keep the approval reference. Retries share the original deadline and storage root. A later reproduction is a new compute allocation and needs its own approval.

Prepare the query workload locally without model imports:

```sh
uv run --locked python scripts/prepare_m8_workload.py \
  --development "$PREPARED/development.parquet" --audit "$PREPARED_AUDIT" \
  --output .local/m8-v1/workload.json
uv run --locked python scripts/analyze_m8_quality.py \
  --development "$PREPARED/development.parquet" --audit "$PREPARED_AUDIT" \
  --m4-report "$M4_REPORT" --m6-report "$M6_REPORT" --per-query "$M6_PER_QUERY" \
  --output .local/m8-v1/saved-quality.json
```

`PREPARED` points to the verified prepared data directory and `PREPARED_AUDIT` to its pinned audit file. The reports and per-query inputs are the immutable files indexed in [directory.md](directory.md). Both commands refuse to overwrite their outputs. The second command reads only the development query columns and saved metrics. It verifies the saved metric hashes and all 4,129 frozen query-group identities.

Transfer the reviewed source snapshot and private workload/approval metadata into a new versioned directory under the existing persistent runtime root. Preserve each source file's SHA-256 in `source/snapshot-manifest.json`. Reuse `env-v2/bin/python`, `artifacts/` and the reconstructed reference recipe described in [M7_DSMLP.md](M7_DSMLP.md). No environment installation or artifact copies are required.

Use the official DSMLP launcher with `-c 4 -m 32`, no GPU option, and the existing pinned notebook image. Save the dry-run and admitted pod specifications. The official launcher requests 2 CPUs/16 GiB and sets limits of 4 CPUs/32 GiB; this is a quota rather than exclusive cores. Submit the supervisor as the container command with `PYTHONPATH` set to the transferred source directory, user-site disabled, offline Hugging Face/Transformers flags, empty `CUDA_VISIBLE_DEVICES`, and all native math/Polars thread settings fixed to 1. Set the pod deadline below the approved cumulative deadline. Never execute the following command on the login node; run it as the allocated container's command:

```sh
"$RUNTIME/env-v2/bin/python" "$M8/source/scripts/run_m8_benchmark.py" \
  --execution-manifest "$M8/inputs/execution.approved.json" \
  --descriptor "$RUNTIME/artifacts/.local/m7-v1/serving-descriptor-v2" \
  --artifacts "$RUNTIME/artifacts" \
  --reference-recipe "$M8/inputs/reference-recipe.json" \
  --workload "$M8/inputs/workload.json" --source-root "$M8/source" \
  --budget-root "$M8" --output "$M8/run-v1"
```

Here `RUNTIME` is the existing M7 persistent root and `M8` is its new versioned benchmark directory. The supervisor requires a Kubernetes compute container, performs resource and recipe admission, validates source/workload bytes, starts one model server and launches separate load-client processes. It exits after writing checksums and shutdown evidence. Logs, query hashes, generated JSON and pod metadata remain private.

After the pod terminates, copy the complete run directory and termination metadata back using the authenticated SSH connection. Verify the complete evidence before summarizing:

```sh
uv run --locked python scripts/analyze_m8_benchmark.py \
  --run-dir .local/m8-v1/run-v1 --workload .local/m8-v1/workload.json \
  --output .local/m8-v1/benchmark-analysis.json
```

Retain the exact submitted command/specification, source snapshot, approved manifest, workload, checksums and failed submission logs alongside this return. Independently compare the returned source-manifest hash against the transferred snapshot, and count every final file in the remote M8 storage root against the approved cap. Capture the terminated pod state before deleting only that owned pod; preserve persistent evidence and original M7 artifacts.
