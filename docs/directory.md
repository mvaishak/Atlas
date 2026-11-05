# Private artifact directory

Inventory recorded 2026-11-04. These paths describe private local state. Generated JSON, raw returns, datasets, model/index files and archives are excluded from the public source checkout. Public documentation reports their results and hashes. A source-only clone can run deterministic fixtures and requires separately supplied private inputs for the real-model procedure.

 Paths are relative to the repository root. This is a map of private local evidence, not permission to rerun historical jobs. Preserve existing files and use new output directories.

The `.local` metadata sweep counted files and byte sizes without opening model weights, raw data or test judgments. Totals exclude symlinks and dependency directories (`venv`, `.venv`, `node_modules`, `.git`). Presence doesn't establish integrity. Specific validations are recorded below.

## Milestone map

| Step | Local location | Contents and evidence status |
| --- | --- | --- |
| M1 | `.local/mlflow/`, `.local/M1-tracking-evidence.json` | Local tracking database/artifact round trips. Private runtime state. |
| M1–M4 checks | `.local/validation/` | 179 files, 2,092,157 bytes at sweep time. Historical command logs, reports and build evidence. |
| M2 prepared data | `data/esci/prepared-v4/` | Canonical catalog, train/development Parquets, query groups, audit and completion marker. Only train/development loaders are authorized. |
| M3 lexical index | `data/esci/indexes/m3-bm25-v2/` | Frozen BM25 backend, products, manifest and completion marker. Bundle `41e8286b…a6a7d`. |
| M4 model | `data/esci/models/all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41/` | Original pinned pretrained model/tokenizer files. |
| M4 full native bundle | `data/esci/infrastructure/m4-full-local-v1/attempt-bd83a1bac3b54cdab48ca32e79f0bf90/bundle/` | Complete FAISS index, vectors, products and sealed metadata. Bundle `b898140c…37a2`. Use this full copy, not the report-only return directory. |
| M4 development baseline | `data/esci/reports/m4-cpu-eval-v1/returned/development/` | Frozen report, completion marker and full per-query rankings/metrics. Source hash checked during M6 statistical replay. |
| M4/M5 transfer history | `.local/colab/` | 1,419 files, 6,847,749,456 bytes at sweep time. Multiple archives/snapshots and returned evidence. Names identify attempts. Never assume the last filename is the accepted run. |
| M5 final checkpoint/model | `.local/colab/m5-training-return-v1/` | `checkpoints/3c8bfa95e4af4447b11784af387fb2c9/step-000574.pt`. Model at `model-bundle/model/`. Earlier verification is in `reviews/M5.training.md`. |
| M5 manual sources | `.local/evaluate_m5_track_a.py`, `.local/export_m5_bm25.py`, `.local/m5-shared-chat-extracted.json`, `.local/m5-shared-chat.html` | Historical adapter sources and supplied manual execution context. Preserve. |
| M5.5 closeout | `.local/m5_5/`, `results/m5_5/`, `configs/m5-reproduction.json` | 26 private log files, 186,115 bytes. Evidence inventory, numeric replay and reproduction pins. Full historical M5 indexes/chunks/rankings remain external. |
| M6 saved demo | `.local/m6/comparison-v1.json`, `.local/m6/actual-smoke-v1.json` | Private five-query comparison and actual BM25/demo smoke evidence. Browser/theme captures also live in `.local/m6/`. |
| M6 shared parity | `.local/m6-parity/` | 72 files, 320,396 bytes. Fixture parity/static logs and preserved Mac native failure. |
| M6 synthetic Linux gate | `results/m6_native_boundary/`, `.local/colab/m6-cpu-boundary-return-v1/` | Small reports plus full 42-case returned synthetic evidence. Original ZIPs remain in `.local/colab/`. |
| M6 preflight | `.local/m6-ranker-prerequisites*-v*/`, `.local/atlasrank-m6-ranker-prerequisites-v1.zip` | Historical native-free byte inventories and blocked runtime package. These are superseded operational states, retained as history. |
| M6 environment preparation | `.local/m6-ranker-execution-v1/` | 401 files, 1,052,502,999 bytes. Source/input archives, receipts, preserved preflight and returned environment/objective-sanity evidence. These precede the user's completed manual fit. |
| M6 final supplied return | `.local/m6/drive files/` | `SUMMARY.md`, `report.json`, `per-query (1).jsonl`, `sealed-model-v1/{manifest.json,COMPLETE.json,model.ubj}`. Returned bytes and saved A/C metrics verified on 2026-11-04. |
| M7 current execution | `.local/m7-v1/` | Initial diff/status/HEAD, saved-ranking replay scripts/logs/JSON, test/static logs and versioned serving descriptors. Includes passing Docker build/fixture smoke and native model loading. The later DSMLP actual-model startup/parity passed, with evidence in the separate return directory below. |

`.local/m6/` held 585 files / 64,327,205 bytes including the supplied return. `.local/builds/` held 24 files / 1,532,818 bytes. Miscellaneous root repair scripts, nested-template JSON and older evidence archives are historical. Don't execute them automatically.

## Supplied M6 hashes

| File | SHA-256 |
| --- | --- |
| `sealed-model-v1/model.ubj` | `91b91e61d731c59bba4f0a3964b2aeeacae20a02aefe4bd91643da69060ea19c` |
| `sealed-model-v1/manifest.json` | `c513845a9d2b059351a84aab054d53c71452d8d80a1bc00f181f9a92074459ab` |
| `per-query (1).jsonl` | `d38f489826c30c54418ad1d7f07cbe0f3f18798872b5af95e5cc79cfbcea4b17` |
| `report.json` | `b267d4f78b19f0e8ba402c6f3e728600b4aeb3704a68aa27d33a4b71dc05dc40` |

The sealed ranker artifact ID is `80903421a685bc848ec1106b059df4f9f5662d3b5cc1aecd270fabb6be8a515f`.

## Historical evidence gaps at the initial inventory

M6's final executed notebook/source delta, real-encoder/fitted-model HTTP parity captures, original fit-result/training-matrix receipts and inference-environment/query-recipe JSON weren't included in the supplied return. The original input manifest is explicitly recorded as lost after a runtime reset. Recovered metadata doesn't recreate that original record.

M5's complete original catalog chunks, native index and full dense/hybrid rankings remain in private Drive according to the M5.5 review. Their historical hashes remain reported rather than freshly checked here. RTK.md and PROCESS.md are still missing from the available instruction paths.

## Historical DSMLP preparation, 2026-11-04

`.local/m7-dsmlp-audit-v1/` contains the local search inventory (2,851 files / 34 ZIP inventories / 15 notebooks), canonical recipe-candidate hashes, byte-level transfer inventory, null-separated allowlists, pinned-image metadata and verified amd64 image archive. The archive is a new local export, not a rebuilt or published image. Originals and `.local/m7-v1/` evidence remain unchanged.

The exact minimal runtime payload is 31 files / 4,121,606,160 bytes. The image archive is 673,882,112 bytes. Detailed missing-input findings, no-login-compute rules, image-delivery blockers and conditional commands are in [M7_DSMLP.md](M7_DSMLP.md). No cluster jobs or transfers had occurred at that preparation checkpoint.

## Recovered M6 records, 2026-11-04

The user subsequently supplied executed notebooks in `Colab Notebooks/`. `xgboost.ipynb` contains checkpointed feature generation, the single fixed fit and evaluation recovery. `Untitled3.ipynb` contains successful final evaluation/sealing output and finalization exit code 0. Notebook source was inspected as historical evidence.

Seven recovered original files are now in `.local/m6/drive files/recovered/`. Their original Drive root is `My Drive/AtlasRank/m6/`.

| Local filename | Original relative Drive path |
| --- | --- |
| `runtime-pins-v1.json` | `recovery-v1/runtime-pins-v1.json` |
| `requirements-installed.txt` | `recovery-v1/requirements-installed.txt` |
| `evaluate_m6_checkpointed.py` | `recovery-v1/evaluate_m6_checkpointed.py` |
| `fit-result.json` | `single-fit-v1/fit-result.json` |
| `FIT_STARTED.json` | `single-fit-v1/FIT_STARTED.json` |
| `m6-training-complete.manifest.json` | `training-feature-checkpoints-v1/m6-training-complete.manifest.json` |
| `atlasrank-m6-source.tar.gz` | `atlasrank-m6-source.tar.gz` |

Read-only audit results and all seven file hashes are in `.local/m7-dsmlp-audit-v1/recovered-m6-audit.json`. Seven metadata/byte-consistency checks passed, including the recovered fit report's model SHA against the actual sealed model. The training matrix itself was not supplied or hashed in this check. Of 79 archived Python source files, 78 match the current checkout. The serving app differs, while encoder and experiment source match. The archive was inspected without extracting or executing its code.

The recovered final runtime records Polars 2.1.0 and PyArrow 26.0.0, both included in the encoder query recipe. The existing M7 image has Polars 1.35.2 and PyArrow 21.0.0. FastAPI and psutil also differ. The full original M6 query-recipe object and actual real-model HTTP parity captures remain missing. Historical M4 recipe objects in the source archive do not fill that gap. This follow-up supersedes the earlier missing-notebook/source/fit-receipt inventory above.

## Authorized recipe reconstruction, 2026-11-04

`.local/m7-dsmlp-audit-v1/reconstructed-m6-query-recipe-v1.json` is newly reconstructed metadata, with an exact canonical match to sealed query digest `a9ad487479249f1f86124025cbf0c7724c883c09f3030d2658ec169b6539b88a`. The companion `reconstructed-m6-query-recipe-v1.provenance.json` records its derivation and original input hashes. `reconstruct_m6_recipe.py` reproduces the derivation and refuses to overwrite existing outputs. `recipe-reconstruction-verification.json` records independent verification, four rejected mismatch cases and original-file/model hash preservation. No native model execution occurred.

The authenticated platform is `Linux-6.6.159+-x86_64-with-glibc2.39`. The user subsequently reported DSMLP kernel `6.1.0-51-amd64` and architecture `x86_64`. `dsmlp-kernel-compatibility.json` records that user-supplied metadata and the failed exact-kernel comparison. The node failed exact historical platform identity. The later explicitly authorized platform policy permitted the isolated DSMLP run while preserving both identities. The original full M6 recipe file and historical HTTP captures remain unavailable. At that checkpoint, the prepared transfer script excluded the new recipe and provenance. The later update included both. Current compatibility findings and the conditional image correction are documented at the top of `docs/M7_DSMLP.md`.

## Portable DSMLP execution update, 2026-11-04

The user subsequently directed execution on the existing DSMLP container after the compatibility-policy discussion and reported the initial transfer complete. `.local/m7-dsmlp-audit-v1/m7-update-v2.tar.gz` contains 100 update files, including the seven returned originals, current code and portable acceptance harness. `m7-update-v2-package.json` records its size and SHA. `upload-update-v2.sh` transfers only this small archive and `start-m7-v2.sh`. The latter starts `scripts/run_m7_dsmlp.sh` inside the allocated container. The historical Termius procedure is retained in `docs/M7_DSMLP.md`.

The original and actual recipe digests are separate identities under this explicitly opted-in policy. Every non-platform recipe field remains exact. The M7 resource and HTTP parity criteria are preserved. At that checkpoint, native-environment setup and model execution were pending. The later verified return below closes the technical gate. Existing image/harness evidence and all immutable model artifacts are preserved.

## Verified DSMLP return, 2026-11-04 Pacific

`.local/m7-dsmlp-return-v2/20261105T040832Z-346/` contains the returned `real-http-v2-*`, `setup-v2-*` and `execution-inputs/` directories. The smoke directory contains three full case JSONs, worker/preflight/supervisor results, worker log and seven-file checksum manifest. Setup contains package versions, resource and immutable-input/update verification, setup log and exit status. Execution inputs contain the approved manifest and reconstructed recipe/provenance.

`.local/m7-dsmlp-audit-v1/verify_returned_m7.py` performed standard-library hash and exact JSON comparisons without native model execution. `docs/reviews/M7.dsmlp-acceptance.validation.json` records the verified results and hashes of every returned file. The real-model HTTP gate passed in 58.17 seconds with 5.52 GiB sampled peak RSS, 4 CPUs, 32 GiB allocated RAM and clean shutdown. See `docs/reviews/M7.md` for scope, provenance-policy disclosure and remaining historical/metadata limitations. The user accepted M7 on 2026-11-05.

## M8 reference profiling, 2026-11-05

The user accepted M7 and approved the bounded CPU benchmark in [M8_PLAN.md](M8_PLAN.md). Private operator files are under `.local/m8-v1/`. This work reuses the existing remote M7 runtime; it does not copy indexes, replace dependencies or remove earlier evidence.

| Path | Contents |
| --- | --- |
| `.local/m8-v1/workload.json` | Hash-selected 1,000-query development workload, source/audit pins and exclusion counts. Contains query text; keep private. |
| `.local/m8-v1/saved-quality.json` | Corrected saved M4/M6 metric replay and whitespace-token error slices over 4,129 query groups. No new quality inference. |
| `.local/m8-v1/saved-quality.analyzer-slices.v1.json` | Preserved intermediate analysis using analyzer-token slices; superseded by the corrected report. |
| `.local/m8-v1/remote-tree/` | Exact transferred source snapshot and private approved inputs. Source-manifest hash is recorded in the M8 review. |
| `.local/m8-v1/submission-attempt-*.log`, `cluster-run.log` | First admission-webhook timeout and subsequent official-specification submission. |
| `.local/m8-v1/admitted-pod.json` | Actual image digest, limits/requests, command and cluster admission metadata. |
| `.local/m8-v1/live-cluster.log`, `progress-summaries/`, `early-results/` | Provisional observations while the benchmark is active; these do not substitute for the complete checksummed return. |
| `.local/m8-v1/final-*.log`, `dist/` | Local synthetic/static/package verification. Generated archives remain private. |

The remote versioned directory is `atlasrank-m7-dsmlp-v1/m8-v1/` beneath the user's persistent home. `source/` and `inputs/` are immutable execution inputs; `run-v1/` holds the actual benchmark output. Each trial records `warmup.jsonl`, `requests.jsonl` and `summary.json`. Top-level evidence includes `preflight.json`, `environment.json`, `server-ready.json`, `server-shutdown.json`, `result.json`, logs and a complete `checksums.json`. The complete return is now in `.local/m8-v1/returned-root/`, including unchanged `source/`, approved `inputs/`, submitted pod metadata and `run-v1/`. `.local/m8-v1/benchmark-analysis.json` verifies all 27 raw trials; `return-validation.json` binds source/workload/dependency identities, cumulative time, final storage, model hash and pod termination/cleanup. `installed-environment.json` and its validation receipt verify all 56 pins; `terminated-pod.json` and `pod-cleanup.log` preserve exit 0 and deletion of the owned pod. Final apparent size, including directories, is 21,533,372 bytes; allocated blocks are 9,947,648 bytes. The 89 run-file checksums and 93 source-file hashes passed. M8 lead acceptance remains pending. This directory's complete byte count, including failed submissions and final receipts, is charged against the approved new-storage cap.
