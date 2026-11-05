# M7 DSMLP CPU acceptance and execution history

## Verified execution result, 2026-11-04 Pacific

The user completed run `20261105T040832Z-346` and returned the evidence. Local hash/metadata/capture verification passed. All three real-model search cases and three rerank cases passed exact parity. Startup to readiness was 49.645922 seconds, supervised wall time was 58.168239 seconds, sampled worker-tree peak RSS was 5.524242 GiB and reported process high-water RSS was 5.527264 GiB. Worker exit was 0 and HTTP shutdown completed. The remaining real-model HTTP technical gate is verified. The user accepted M7 on 2026-11-05.

The actual environment was Python 3.12.11, CPU Torch 2.14.1+cpu and the recovered dependency pins on Linux `6.1.0-51-amd64`, glibc 2.35, with 4 effective CPUs and 32 GiB cgroup memory. Model-admission effective available RAM was 28,225,970,176 bytes. Original and actual recipe identities are separately recorded. Original sealed bytes remain unchanged. This was the isolated native-environment route, separate from the old pinned Docker image tests.

Results and limitations are in [M7 review](reviews/M7.md) and `docs/reviews/M7.dsmlp-acceptance.validation.json` (private generated evidence). Full returned data is under `.local/m7-dsmlp-return-v2/20261105T040832Z-346/`. Subsequent sections retain preparation history and their then-current unrun status.

## Source-only reproduction

The public source includes fixture tests, notebook templates, the Docker recipe, DSMLP preflight/parity scripts and `environments/dsmlp-cpu/requirements.txt`. The real-model run requires privately held immutable components, reconstructed recipe/provenance and an explicitly approved execution manifest. Paths below document local evidence and earlier commands. Private artifacts, generated validation JSON, executed notebooks and transfer archives stay in private storage.

The recorded DSMLP environment used the recovered package pins, including Polars 2.1.0, PyArrow 26.0.0, FastAPI 0.162.0 and psutil 7.2.2, plus M7's Uvicorn 0.54.0. The earlier Docker fixture environment used different versions and source bytes. Its checks establish the behavior of that recorded image. See [the serving guide](M7_SERVING.md) for fixture commands and [the review](reviews/M7.md) for the separately verified real-model run.

The original full M6 recipe and historical HTTP captures remain unavailable. The reconstructed reference canonically matches the immutable historical digest. The actual DSMLP digest records its kernel/glibc difference under the explicit compatibility policy. A new real-model run needs its own authorization, resource checks and returned evidence.

## Prepare a new source package

Package generation is a local metadata operation. Supply the private reference recipe, its provenance, the immutable serving descriptor and the intended execution manifest explicitly. The packager validates the reference against the sealed pin and copies the manifest verbatim. Execution approval remains a separate decision.

```sh
.venv/bin/python scripts/package_m7_dsmlp.py \
  --reference-query-recipe /private/inputs/reconstructed-m6-query-recipe-v1.json \
  --reference-provenance /private/inputs/reconstructed-m6-query-recipe-v1.provenance.json \
  --descriptor /private/inputs/serving-descriptor-v2 \
  --execution-manifest /private/inputs/m7-execution.json \
  --output /private/outputs/m7-source-new.tar.gz
```

Replace `/private/inputs` and `/private/outputs` with your verified local paths. The new archive contains current source, the reviewed requirements and explicitly supplied metadata under `update-v2/`. The descriptor and runtime components stay outside it. Its source hashes identify a new package, separate from the 100-file historical update.

On an authorized allocated Linux CPU container, stage the immutable component tree under `artifacts/` and its verified byte inventory at `metadata/transfer-inventory.json` beside the extracted `update-v2/`. Keep the descriptor at `artifacts/.local/m7-v1/serving-descriptor-v2`, as required by the runner. The runner expects `/opt/conda/bin/python` as its bootstrap interpreter and creates an isolated Python 3.12 environment. Preserve `evidence/setup-v2-*` and `evidence/real-http-v2-*`, including failures. File transfer, package installation and model execution belong inside the approved workflow. Run compute work only in the allocated container.

The runner defaults to exact platform identity. To reproduce the documented DSMLP platform exception, the approved manifest must declare `"compatibility_policy": "record_linux_platform_difference"`, and the operator must pass the flag explicitly from the persistent run root.

```sh
bash update-v2/scripts/run_m7_dsmlp.sh --allow-platform-difference
```

## Historical procedure scope

Everything below records preparation checkpoints from 2026-11-04 before the verified result. Missing inputs, unrun statements, exact-platform stop conditions, approval flags and private helper commands describe those checkpoints. The later approved isolated environment and platform policy superseded that execution route. The private upload/start/fetch scripts are retained as evidence of the executed handoff.

## Historical execution handoff, 2026-11-04

Following the platform discussion, the user directed proceeding on the existing DSMLP container and reported the bulk transfer complete. The new portable execution route uses an isolated native Python 3.12 environment with recovered CPU dependency pins. Linux kernel/glibc identity is recorded as actual runtime provenance. Every other recipe field, including dependency versions, source, model files and scoring settings, remains exact. The sealed model, serving descriptor and historical recipe digest retain their original bytes. Strict loading remains the default. The M7 runner opts into the new policy explicitly in both its manifest and command flag. This supersedes the earlier exact-platform stop condition for this authorized M7 run.

On the local Mac, run `bash .local/m7-dsmlp-audit-v1/upload-update-v2.sh`. It transfers the 922,920-byte update archive and its startup script through the existing private SSH route. The update includes current source, the runner, all seven newly returned originals, reconstructed recipe/provenance and exact CPU requirements. The old image and transferred harness remain preserved as earlier evidence.

In the existing allocated Termius container, run `transport/start-m7-v2.sh` beneath the verified `atlasrank-m7-dsmlp-v1` root. It verifies the archive checksum, extracts into a fresh `update-v2/`, and invokes `scripts/run_m7_dsmlp.sh`. Setup verifies all 31 immutable inputs and update-file hashes, records resources, creates separate bootstrap/Python environments, installs CPU Torch and the pinned package set, and checks dependencies. The recovered snapshot's unused CUDA-related packages are excluded. Uvicorn 0.54.0 is retained from M7. Setup time is logged separately from the smoke.

The supervised smoke retains 4 CPUs / 32 GB / 0 GPUs, >=8 GiB effective available RAM, <=12 GiB sampled/high-water RSS and the 300-second ceiling including shutdown. It runs actual real-model HTTP search/rerank cases and exact feature/prediction comparisons. New evidence is written under timestamped `evidence/setup-v2-*` and `evidence/real-http-v2-*`. Original/runtime query recipe IDs and full actual runtime recipe are recorded separately. Failed output is retained and stops the script.

Local validation passed 67 targeted tests, Ruff and strict mypy across 139 source files, shell/embedded-Python syntax, all 100 update-file hashes and a separate read-only compatibility review. Results are recorded in `.local/m7-dsmlp-audit-v1/portable-update-validation.json`. At this checkpoint, dependency installation and real-model execution were pending. The verified result above supersedes that status. No new cluster allocation, Mac neural/native run, model fit, evaluation or commit occurred.

Historical reconstruction checkpoint, 2026-11-04. The local metadata reconstruction exactly matches the sealed M6 query digest. This is reconstructed evidence. The original full JSON and historical real-model HTTP captures remain unavailable. The execution proposal remains `approved: false`. See the current amendment below before using the historical procedure. The SSH authentication probe failed, and this agent has performed no bulk transfer or cluster submission.

## Historical amendment before the approved platform policy

The user authorized reconstruction after the seven original recovery files arrived. The resulting `.local/m7-dsmlp-audit-v1/reconstructed-m6-query-recipe-v1.json` has canonical digest `a9ad487479249f1f86124025cbf0c7724c883c09f3030d2658ec169b6539b88a`, exactly equal to the sealed manifest's query pin. Its companion `.provenance.json` records the inputs, hashes and reconstruction method. Keep both files together in a separately named `reconstructed-records/` staging directory when a revised transfer is prepared. The transfer script at this checkpoint did not include them.

The reconstruction reads literal recipe fields from the archived encoder AST, verifies all eleven pinned model files, and uses package versions agreed by both recovered runtime records. The platform string from the historical M4 CPU recipe was tested as one candidate. Exact canonical digest equality confirms that candidate's complete content against the sealed M6 pin. This does not recover the missing historical capture or establish where/when a capture was written. The existing preflight's `--original-query-recipe` flag can validate the raw reconstructed recipe mechanically, but its name and evidence keys require explicit provenance clarification before approving any run with this reconstruction.

The derivation command was `python3 .local/m7-dsmlp-audit-v1/reconstruct_m6_recipe.py` from the repository root. The script uses exclusive writes and deliberately refuses a rerun over existing outputs. It imports no AtlasRank or ML modules and executes no archived code. Independently checked canonical hashing and the existing preflight comparator accept exact metadata and reject changed platform, packages, source hash and recipe content. Results are in `.local/m7-dsmlp-audit-v1/recipe-reconstruction-verification.json`. All seven recovered file hashes and the sealed model hash were rechecked.

### Platform requirement comes before an image build

The authenticated recipe requires `Linux-6.6.159+-x86_64-with-glibc2.39`. In the existing allocated Termius compute container, obtain `uname -r` and `uname -m`. The required outputs are `6.6.159+` and `x86_64`. The current container's glibc does not determine a future image's glibc, while the host kernel remains shared by containers. If the kernel differs, an image rebuild cannot satisfy this exact recipe. Stop and retain the mismatch. Never spoof platform metadata or reseal the model.

The user subsequently returned `6.1.0-51-amd64` and `x86_64` from the existing compute container. Architecture matches, but the kernel differs. This is user-reported environment evidence, recorded separately in `.local/m7-dsmlp-audit-v1/dsmlp-kernel-compatibility.json`. The current DSMLP node fails the frozen platform requirement before model execution. Image correction and additional payload transfer cannot resolve this host-kernel mismatch.

Under the existing criteria, the next acceptance attempt requires an authorized host that actually reports kernel `6.6.159+` and architecture `x86_64`, with the corrected glibc/package environment. Original Colab CPU is a candidate to investigate because the matching platform string came from historical Colab evidence. Its current kernel and capacity remain unverified. A new allocation needs user approval, followed by a metadata/resource check before any model load. Reusing an already-running authorized CPU runtime permits the same bounded metadata check without a new allocation. If no compatible host exists, stop for an explicit architecture decision. The present scope grants no compatibility-contract change.

### Proposed image correction, pending platform check and approval

| Component | Existing M7 image | Required recovered M6 value |
| --- | --- | --- |
| Architecture | Linux amd64 | Linux amd64 |
| glibc | 2.36 | 2.39 |
| Torch CPU | 2.14.1+cpu | 2.14.1+cpu |
| Sentence Transformers | 6.1.0 | 6.1.0 |
| Transformers | 5.20.2 | 5.20.2 |
| tokenizers | 0.23.3 | 0.23.3 |
| safetensors | 0.8.0 | 0.8.0 |
| NumPy | 2.5.4 | 2.5.4 |
| Polars | 1.35.2 | 2.1.0 |
| PyArrow | 21.0.0 | 26.0.0 |
| FastAPI | 0.156.4 | 0.162.0 |
| psutil | 7.1.3 | 7.2.2 |

Retain XGBoost CPU 3.1.1, FAISS CPU 1.15.1, BM25S 0.2.14 and HTTPX 0.28.1. Use the recovered package snapshot to review transitive dependencies, including Starlette 1.10.0. Retain the existing M7 Uvicorn 0.54.0 serving dependency, which is absent from the recovered M6 snapshot, and declare this addition to the recovered environment explicitly. Python remains 3.12. The fit marker records 3.12.3, while the final recovery pins specify only 3.12. Python patch version is outside this recipe's hash and must still be recorded accurately for a new run.

A compatible base must be verified and pinned by immutable amd64 digest, with glibc 2.39 confirmed from its actual bytes. No new base image has been selected or pulled. Prepare a separate fully locked image/requirements set and fresh image identity after approval, leaving the original Dockerfile, lockfile, image export and historical results intact. Original installed CUDA-related distribution entries remain in the recovered snapshot as history. Do not infer GPU use from those entries or blindly add them to a CPU image. Any excluded transitive distributions must be reviewed and recorded. Use CPU Torch and allocate zero GPUs.

Two execution alternatives remain. A DSMLP node with the exact authenticated kernel permits a separately approved compatible image and the existing bounded acceptance procedure. If DSMLP's kernel differs, an execution environment with the exact required kernel must be identified under separate resource approval. A portability redesign would change the compatibility contract and requires a separate architecture decision beyond this execution plan.

Failure checks remain exact recipe/source/model compatibility, complete post-DMatrix HTTP parity, 8 GiB effective available RAM, 12 GiB peak RSS, 4 CPUs / 32 GB / 0 GPUs, the 300-second limit and verified shutdown. Metadata reconstruction does not establish real HTTP acceptance. At this checkpoint, image build/import, revised provenance-aware staging and actual execution were unrun.

## 1. Findings and stop conditions

The following inventory records the initial audit before the notebook/recovery-file return and the authorized reconstruction. The current amendment supersedes its missing-input status and its prohibition on creating a separately labeled reconstruction. The original-evidence restriction remains in force.

The local audit enumerated 2,851 files under `.local`, `notebooks` and `results`, inspected 34 ZIP and 12 historical source TAR inventories, and summarized 15 notebooks. Metadata searches were bounded to 4 MiB text files / 1 MiB archive metadata members. Original raw datasets, credentials, binary checkpoints and test judgments weren't searched. Relevant frozen M4 report metadata was inspected separately. Detailed search evidence is retained under `.local/m7-dsmlp-audit-v1/`.

| Requested input | Result | Meaning for M7 |
| --- | --- | --- |
| Final executed real-model M6 notebook / source delta | Not found | The 3 M6 notebook copies are unexecuted synthetic-boundary templates. Preserve this historical gap. Don't rename a new notebook as the original. |
| Actual encoder + fitted-model post-DMatrix HTTP captures | Not found | Available native-boundary captures are synthetic zero-update checks. M7 must produce new real-model evidence, explicitly dated as M7. |
| Original full M6 CPU query-recipe JSON | Not found | The sealed model contains digest `a9ad487479249f1f86124025cbf0c7724c883c09f3030d2658ec169b6539b88a`, not the full recipe. Restore a genuine source record before the proposed metadata preflight. |
| Original M6 final environment JSON | Not found | Earlier provisioning `pip-freeze.txt`, install reports and native smoke records exist, but precede final manual completion. They aren't the missing final query-recipe record. |
| Full older M4 CPU query recipe | Found in the frozen M4 report and `.local/m6/comparison-v1.json` | Its digest is `d9328de54c74068ab861721f4da5b668de99edf4c012801e45a5a80f68d2d86b`, not M6's pin. Never substitute it. |
| M6 final sealed model/report/per-query rankings | Present and previously verified. Component hashes rechecked | Model can transfer unchanged. Evaluation outputs stay local because the smoke doesn't use judgments or recompute development metrics. |
| Original input manifest / final fit-result / training-matrix receipt | Input manifest recorded lost. Final receipts not found | Keep as M6 historical limitations. No fitting, feature-matrix recreation or historical-budget claim is part of this plan. |

Twenty-one complete recipe objects found in relevant metadata were canonically hashed. None matched the sealed M6 query digest. The complete M4 product recipe does match its original product pin. This separates a missing M6 query record from the available pretrained artifacts.

RAM alone doesn't make this migration executable. The query digest includes `platform.platform()`, `platform.machine()`, runtime versions and encoder source hash. Containers use the host kernel. Even identical image bytes can fail this digest on a DSMLP kernel. Recovering the original JSON allows the exact comparison. It doesn't authorize changing either side. **A mismatch stops execution.** No platform spoofing, digest override or compatibility waiver is proposed.

Historical missing notebook/captures don't become original evidence after a new run. A successful new M7 run can close its current serving gate while retaining those M6 history gaps, provided the frozen compatibility checks pass.

## 2. Immutable transfer set

The existing serving descriptor already uses relocatable paths beneath an artifact root. Preserve that relative tree. Transfer the descriptor bytes unchanged instead of resealing it with a different recipe.

| Component | Files | Bytes | Repository-relative source |
| --- | ---: | ---: | --- |
| M4 dense bundle | 5 | 3,808,634,983 | `data/esci/infrastructure/m4-full-local-v1/attempt-bd83a1bac3b54cdab48ca32e79f0bf90/bundle/` |
| M3 lexical bundle | 8 | 221,212,561 | `data/esci/indexes/m3-bm25-v2/` |
| Original pretrained encoder/tokenizer | 11 | 91,578,415 | `data/esci/models/all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41/` |
| Sealed M6 ranker | 3 | 176,443 | `.local/m6/drive files/sealed-model-v1/` |
| Dense configuration | 1 | 2,333 | `configs/dense.json` |
| Existing serving descriptor | 3 | included below | `.local/m7-v1/serving-descriptor-v2/` |

Total: **31 files, 4,121,606,160 bytes (about 3.84 GiB)**. Every component file was streamed through SHA-256. Manifest pins and model-file inventory match. The native index wasn't loaded during this audit. Full native reconstruction remains inside the real startup check.

The runtime/harness allowlists, sizes and hashes are `.local/m7-dsmlp-audit-v1/transfer-inventory.json`, `artifacts.files0` and `harness.files0`. Null-separated lists preserve the space in `drive files`. The generated `code.files0` stays local as an optional provenance inventory. A second code checkout is unnecessary because the exact image already contains all 82 matching source files. Python must import that installed package, never a host `.venv` or replacement source tree.

Do not transfer raw ESCI files, train/development/query-group Parquets, the canonical catalog Parquet, M5 artifacts, training matrices/checkpoints, raw query logs, credentials, the full `.local` tree, or Git history. The loader consumes label-free products already inside the immutable lexical/dense bundles. Keep original directories untouched and use a new private persistent run directory. Plan for at least 12 GiB free persistent storage for artifacts, image archive, code/evidence and headroom. Confirm the actual quota and mounts first.

## 3. Exact image and dependency audit

The existing image is Linux/amd64, about 674 MB, Python 3.12.11 and glibc 2.36. It uses CPU Torch 2.14.1+cpu, Sentence Transformers 6.1.0, Transformers 5.20.2, tokenizers 0.23.3, safetensors 0.8.0, NumPy 2.5.4, Polars 1.35.2, PyArrow 21.0.0, FAISS CPU 1.15.1, XGBoost CPU 3.1.1, BM25S 0.2.14, FastAPI 0.156.4, Uvicorn 0.54.0 and psutil 7.1.3. HTTPX 0.28.1 is present. Pytest isn't installed. The prepared harness uses the standard HTTP client and existing serving/parity functions, so no packages need installing.

The image's current nonroot user is `atlas`, working directory `/app`, and default entrypoint launches `scripts/serve_bundle.py --container`. DSMLP must override that entrypoint for the controlled preflight/harness. Verify UID, readable persistent mounts and `/app/.venv/bin/python` in the dry-run pod specification. Don't rebuild onto UCSD's notebook base image or inherit its Conda environment.

| Identity | Value |
| --- | --- |
| Previously recorded local image/index ID | `sha256:6cebdd22ab0c98d452771d93cbff8c1a4ba8e205ae134d7e8ba0d0242712cfb7` |
| Exported amd64 platform manifest | `sha256:2abecb0c9798a317286fbeea5d12d77f4dad3897156ec1e5d38e696e3db30fa0` |
| Exact runtime configuration blob | `sha256:3e02f061056b5dfb9cade8f99507a7ebaedd0a8cd6cd9367b4feffad08efb324` |
| Local image archive SHA-256 | `95534c792b54a63462d0255d68ce01b63a9badf0f4d2d8820f172f4a512101e9` |
| Local archive bytes | 673,882,112 |

The already prepared archive is `.local/m7-dsmlp-audit-v1/atlasrank-m7-cpu-amd64.tar`. All 12 content-addressed blobs were checked, and the configuration's ordered rootfs diff IDs and environment equal the inspected original. The archive selects the original amd64 platform image. The surrounding multi-platform/attestation index needn't keep the same digest after import. Verify platform manifest, configuration and layers, rather than confusing the local index ID with a registry manifest reference.

Docker's supported archive route is `docker image save` / `docker image load`. It preserves image content and tags without reinstalling dependencies. A pinned Dockerfile rebuild can preserve dependency versions, but isn't proof of the same image bytes. Use the existing archive. [Docker save](https://docs.docker.com/reference/cli/docker/image/save/), [Docker load](https://docs.docker.com/reference/cli/docker/image/load/).

### Image availability is an unresolved gate

DSMLP launches images through Kubernetes. Its normal custom-image route is a pullable registry reference. A tar file in persistent storage isn't itself launchable with `launch.sh -i`. The UCSD example's GitHub/GHCR publication workflow conflicts with the current no-publication instruction, so it isn't selected. [UCSD custom-image example](https://github.com/ucsd-ets/datahub-example-notebook).

Preferred route under the current constraint: obtain ITS confirmation of an approved **offline import of this existing image** into the compute-node runtime/cache, including the allowed node/pull policy. This is an administrative dependency, not an operation to run on the login node. If ITS offers an approved internal image-delivery service, obtain explicit approval for that external change before using it. A private registry push still needs approval. Docker-in-Docker, privileged pods and Apptainer support aren't assumed.

No registry push, image import, permissions change or ITS message was performed. The local `RepoDigests` field doesn't prove a registry is reachable or contains this image.

## 4. DSMLP allocation and storage discovery

UCSD documents `launch.sh -c`, `-m`, `-g`, custom `-i`, alternate entrypoint `-e`, shell-only `-s` and specification-only `-d`. It prohibits manually running programs on the login node. Our allocation is **4 CPUs, 32 GB RAM, 0 GPUs**, with native math/encoder threads retained at **1** from the frozen recipe. [Launch guide](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032269), [configuration flags](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032273).

The following are prepared commands, **not executed**. After approval, login-node work is limited to authentication, file management/transfer and scheduler metadata/control. No Python, dependency installation, archive extraction, bulk hashing, model loading, server or editor runs there.

```sh
# Local terminal. Replace USERNAME, WORKSPACE and verified paths before use.
ssh USERNAME@dsmlp-login.ucsd.edu

# Login node: metadata and file-management commands only.
workspace --list
launch.sh -h
```

Record the actual persistent host path and its canonical path inside the chosen pod. The loader rejects symlinked ancestors. Course workspaces and personal `/private` mounts differ. Don't guess their equivalence or use container-local `/tmp` for returned evidence. UCSD documents SCP/SFTP/rsync and workspace discovery. [Transfer guide](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032277).

Once ITS has confirmed the unchanged image's launch reference and pull policy, inspect a specification without submitting:

```sh
# Login node, AFTER the image route and account/workspace are known.
# APPROVED_IMAGE_REF and APPROVED_PULL_POLICY must come from the approved route.
K8S_TIMEOUT_SECONDS=600 launch.sh -d -c 4 -m 32 -g 0 \
  -i "$APPROVED_IMAGE_REF" -P "$APPROVED_PULL_POLICY" \
  -J -e /bin/bash -s > m7-pod-proposed.json
```

A `-d` result is a proposed specification, not proof of server admission or node image-cache availability. After an approved launch, also retain the actual admitted pod specification/status. Before any submission, verify its CPU request/limit is 4, RAM allocation is 32 GB, GPU resources are absent, entrypoint is the shell, image/architecture is correct, persistent mounts and UID allow input reads/evidence writes, and `activeDeadlineSeconds` reflects the proposed 600-second allocation ceiling. If the launcher doesn't honor the ceiling or image policy, stop for a corrected approved specification. Don't apply an improvised privileged pod.

## 5. Prepared transfer commands

Use a fresh private host-side directory that maps to the verified persistent container path. Create the root and its `artifacts`, `transport`, `metadata`, `harness`, `original-records`, and `evidence` subdirectories through ordinary file management after approval. Existing run directories must not be reused.

```sh
# LOCAL Mac terminal only. The transfer receiver performs file transport only.
AUDIT=.local/m7-dsmlp-audit-v1
DSMLP_USER=USERNAME
DSMLP_HOST_ROOT=/VERIFIED_PRIVATE_PERSISTENT_PATH/atlasrank-m7-dsmlp-v1
DSMLP_DEST="${DSMLP_USER}@dsmlp-login.ucsd.edu:${DSMLP_HOST_ROOT}"

rsync -rltp --relative --ignore-existing --from0 \
  --files-from="$AUDIT/artifacts.files0" ./ "$DSMLP_DEST/artifacts/"
rsync -rltp --relative --ignore-existing --from0 \
  --files-from="$AUDIT/harness.files0" ./ "$DSMLP_DEST/harness/"
scp "$AUDIT/atlasrank-m7-cpu-amd64.tar" "$DSMLP_DEST/transport/"
scp "$AUDIT/transfer-inventory.json" "$AUDIT/image-archive-verification.json" \
  "$AUDIT/image-environment.json" "$AUDIT/harness-manifest.json" "$DSMLP_DEST/metadata/"
```

The image archive is for the approved import mechanism. Don't run `docker load` on dsmlp-login or assume a Docker daemon exists inside a pod. No `--delete`, full checkout sync, secret-directory sync or remote bulk checksum is part of transfer. `--ignore-existing` protects prior bytes. A subsequent hash mismatch stops rather than silently repairing a reused destination. Record transfer start/end and source/destination manifests.

Restore the genuine original query-recipe record into `original-records/` separately, with source provenance and its file SHA-256. Do not synthesize it from the older M4 recipe, an environment probe or the digest. The original executed notebook/parity records, if later recovered, remain provenance evidence rather than runtime dependencies.

## 6. Inside the approved allocated CPU container

Only after explicit job approval, use the inspected launch options **without `-d`**. Keep the pod foreground and terminate it when done. The proposed outer allocation ceiling is 600 seconds. The actual smoke ceiling remains 300 seconds. Image scheduling/pull and byte transport are setup records, never hidden smoke time.

Inside the allocated container:

1. Confirm hostname isn't a login node, the pod has no GPU allocation and effective cgroup CPU/memory limits match 4/32. Record node/pod/image identities with scheduler metadata. Use `/app/.venv/bin/python -s` and remove `PYTHONPATH`. Ignore mounted user-site packages.
2. Verify all 31 transferred input file sizes and SHA-256 values from `metadata/transfer-inventory.json` using a streaming hash in this compute container. Reject symlinks/escaping paths. Save a fresh `transport-verification.json`. Perform archive extraction/import verification only through the approved compute/admin route. Never hash/extract the multi-GB payload on the login node.
3. Run the metadata-only preflight. It checks effective available memory as the minimum of host available RAM and cgroup limit minus usage. It requires at least 8 GiB, 4 effective CPUs, the 32 GB container limit, and the original recipe's exact digest/runtime/platform/source correspondence. It loads no neural model.

```sh
# COMPUTE CONTAINER ONLY. Set the verified container-side persistent root.
DSMLP_CONTAINER_ROOT=/VERIFIED_CONTAINER_PERSISTENT_PATH/atlasrank-m7-dsmlp-v1
RUN=$(realpath "$DSMLP_CONTAINER_ROOT")
DESC="$RUN/artifacts/.local/m7-v1/serving-descriptor-v2"
RECIPE="$RUN/original-records/original-m6-cpu-query-recipe.json"

# No environment re-resolution or package installation.
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /app/.venv/bin/python -s "$RUN/harness/scripts/m7_dsmlp_preflight.py" \
  --descriptor "$DESC" --original-query-recipe "$RECIPE" \
  --output "$RUN/evidence/preflight-v1.json"
```

Exit code 2 is a preserved blocked preflight, not a successful acceptance run. If the recipe/platform differs or the record is absent, stop and return the evidence. Don't patch `platform.platform`, edit the digest, load a substitute bundle or install matching-looking packages.

4. Following approval, create a **new approved copy** of `configs/m7-dsmlp-smoke.proposed.json`, record the human approval reference and approved image delivery identity, and hash that execution manifest. Preserve the proposal as false. The runner rejects it until `approved` is true and the exact 4/32/0, 8 GiB admission, 12 GiB RSS and 300-second contract is retained.

```sh
# COMPUTE CONTAINER ONLY, after the explicit approved manifest exists.
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /app/.venv/bin/python -s "$RUN/harness/scripts/m7_dsmlp_parity.py" \
  --execution-manifest "$RUN/metadata/m7-execution-approved-v1.json" \
  --descriptor "$DESC" --artifacts "$RUN/artifacts" \
  --original-query-recipe "$RECIPE" \
  --output "$RUN/evidence/real-http-v1"
```

## 7. Real-model HTTP procedure and limits

The prepared supervisor starts a fresh worker process group. The measured smoke begins before component loading: original component hash checks, native index reconstruction/norm/alignment, encoder/model loading, actual HTTP startup/readiness, parity requests and shutdown all fall within the 300-second envelope. It polls worker-tree RSS every 0.1 seconds and also checks the worker-reported process high-water RSS. Any observed value above 12 GiB fails and stops the group. At 290 seconds it stops work, reserving 10 seconds for termination/reporting. Monitoring failure or controlled SIGTERM/SIGHUP/SIGINT interruption also stops the owned process group. The Linux worker binds a parent-death signal before model imports, and cleanup handles a group whose root already exited. No automatic retry is allowed.

The worker reuses `load_runtime`, `offline_predict`, original BM25/dense/RRF, `FittedPredictor`'s native post-DMatrix capture and `assert_exact_parity`. It rejects deterministic or zero-update predictors. It starts the current `create_app(serving_runtime=...)` behind a real loopback Uvicorn HTTP socket, one worker, the existing connection limits and no cache. It doesn't expose the server externally.

The 3 new label-free M7 smoke queries are `wireless headphones`, its fullwidth/whitespace variant, and a declared out-of-vocabulary token. These are new cases, never mislabeled recovered M6 cases. For each query:

- Independently construct the offline BM25 top-100 / dense top-100 / RRF-60 top-100 pool and score it through the original offline entry point.
- POST `/search` selecting `m6_xgb_m4`, `top_k=100`. Compare actual final native captures, predictions, product order and complete candidate membership exactly. Confirm experimental/live/model identities.
- POST `/rerank` with a permuted small real candidate list, including a product outside the independent retrieval lists and an empty-title product if one exists. Compare native feature inputs and predictions exactly. Record genuine computed BM25 zeros. Never fill absent top-100 scores.
- Capture names/order, float32 dtypes, values, missing masks, empty-text flags, canonical row identities, schema/provenance and outputs. Original fixture negative/mismatch cases remain preserved and locally checked. The production image needs no pytest install.

Save each completed case immediately. Check the model hash before/after. Never save new weights. Stop the HTTP thread, verify shutdown, and write the worker result. The parent only marks success when worker exit/result, wall time and RSS checks all pass. Evidence-file hashing happens after the stopped smoke and is timed separately. This is a bounded correctness check, with no relevance metrics, full development evaluation, load workload or optimization variants.

The harness is prepared and locally tested for its helpers/guards/capture serialization. Its real-model worker is **unrun**. Historical fixture/container/model-load successes remain separate evidence.

## 8. Evidence needed for M7 closure

Return a new immutable private evidence directory containing:

1. Approved execution manifest and hash. Pod specification/status, actual image/platform/config/layer identities, node architecture, UID, 4 CPUs / 32 GB / 0 GPUs allocation and outer deadline. Never include auth tokens or registry credentials.
2. Source/harness hashes and 31-file transport verification. Original descriptor/M3/M4/M6 IDs and complete component hashes. Original query-recipe content digest, file hash and provenance.
3. Python/glibc/kernel/architecture, package versions, effective CPU quota/affinity and cgroup memory limit/current/available at admission. Record frozen single-thread environment values.
4. Startup/readiness time and actual readiness response, complete post-DMatrix offline/HTTP matrices and outputs for both routes, all exact comparisons, candidate-membership checks, zeros/missingness/normalization coverage and HTTP version/score metadata.
5. Supervisor elapsed time, sampled peak worker-tree RSS, sampling interval, exit status, shutdown result and partial evidence on failure. Preserve kernel/cgroup OOM evidence if applicable. Sampling can miss short spikes. State that limitation.
6. Model before/after SHA equality, evidence checksums and confirmation of no fits, indexing, training, label access, M8/M9 work or promotion. Preserve the existing 315-pass/8-skip M7-era fixture evidence and Docker smoke records.

A successful lexical smoke, native import, sealed-model load, matching predictions without exact feature capture, or allocated 32 GB alone cannot close the gate. Any failed compatibility/resource check stays a failed or blocked run. Return results before authorizing any retry or changing the plan.

### Decisions still needed before submission

- The genuine final M6 CPU query-recipe record, and any recoverable executed notebook / fitted parity captures.
- DSMLP username/workspace and verified private host/container storage mapping.
- ITS-approved image archive import/delivery mechanism under the no-publication constraint.
- Human approval of the concrete dry-run pod and bounded execution manifest. If the actual DSMLP platform cannot satisfy the original digest, a separate decision is required. This plan grants no waiver.

### 2026-11-04 transfer follow-up

**Recovered-record update.** The seven requested originals have since arrived in `.local/m6/drive files/recovered/`. The detailed inventory is in `docs/directory.md`, with hashes and read-only consistency checks in `.local/m7-dsmlp-audit-v1/recovered-m6-audit.json`. Fit/model/configuration/matrix-receipt metadata agree. The executed final notebook is `Colab Notebooks/Untitled3.ipynb`, following fitting/recovery in `Colab Notebooks/xgboost.ipynb`.

The final runtime pins and saved package snapshot confirm Polars 2.1.0, PyArrow 26.0.0, FastAPI 0.162.0 and psutil 7.2.2. The existing M7 image contains different versions of all four. Polars and PyArrow participate in the encoder recipe digest, so preserving that image alone cannot reproduce the recovered final runtime. Resolve this compatibility finding before selecting an execution image. A dependency/image change needs its existing explicit approval. The original full M6 recipe and real-model HTTP captures remain unavailable. No replacement original recipe was generated, and the prepared image, transport manifest and execution approval flag were not changed.

The user supplied their private account, allocated container hostname, home path, Python `/opt/conda/bin/python` version 3.11.9, and launch command `launch.sh -c 4 -m 32`. They approved selecting the most efficient transfer route. The selected route is resumable rsync over SSH with one reused authenticated connection. The prepared local script is `.local/m7-dsmlp-audit-v1/transfer-vamenon.sh`. Its shell syntax and local manifest/hash checks passed. Bulk transfer remains unexecuted. The noninteractive SSH probe was refused after the Duo authentication notice, so the user must authenticate from their Mac Terminal.

The script creates a private, marked `atlasrank-m7-dsmlp-v1` directory in the SSH account home. Storage mapping means identifying where that same directory appears inside the allocated compute container. Check both `$HOME/atlasrank-m7-dsmlp-v1` and `$HOME/private/atlasrank-m7-dsmlp-v1` for `.atlas-transfer-id` before selecting the runtime root. The actual mapping remains unverified.

The original runner writes `report.json` with top-level `query_recipe` and `environment`, `real-boundary.json` for the real-encoder boundary, and `fitted-boundary.json` for the fitted-model boundary. These are expected runner filenames. Their existence in the manually completed run remains unverified. The supplied final `report.json` is evaluation-only and lacks those recipe/environment fields. `original-m6-cpu-query-recipe.json` was a proposed staging filename, not a recovered original filename. The initial filename gap was later resolved by `Colab Notebooks/Untitled3.ipynb`, as recorded above.

At this checkpoint, transfer approval covered staging the prepared inputs. Image delivery/import and bounded compute execution still required their existing approvals. The reported default Python 3.11.9 differs from the pinned image's Python 3.12.11. The image archive preserves the existing dependency versions, while an approved DSMLP image import route and exact original query-recipe compatibility remain unresolved.


## Compute-pod transport verification command

Run this only inside the approved allocated container, before metadata preflight. It reads label-free runtime inputs, harness files and the image-installed code, writes a new evidence file and doesn't load models. `$RUN` must already be the verified canonical private container path.

```sh
/app/.venv/bin/python -s - "$RUN" <<'PY'
import hashlib, importlib.util, json, sys
from pathlib import Path
run = Path(sys.argv[1]).resolve(strict=True)
inventory = json.loads((run / 'metadata/transfer-inventory.json').read_text())
harness = json.loads((run / 'metadata/harness-manifest.json').read_text())
verified = []
for root, records in [(run / 'artifacts', inventory['component_files']),
                      (run / 'harness', harness['files'])]:
    for record in records:
        path = root / record['path']
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root.resolve()):
            raise ValueError('Unsafe transferred input path')
        if path.stat().st_size != record['bytes']:
            raise ValueError('Transferred input size mismatch')
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != record['sha256']:
            raise ValueError('Transferred input hash mismatch')
        verified.append({'path': str(path.relative_to(run)), 'sha256': digest.hexdigest()})
image = json.loads((run / 'metadata/image-environment.json').read_text())
package = importlib.util.find_spec('atlasrank')
if package is None or package.origin is None:
    raise ValueError('Image-installed package is missing')
installed = Path(package.origin).parent.parent
for name, expected in image['source_sha256'].items():
    path = installed / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Image-installed source differs')
with (run / 'evidence/transport-verification.json').open('x') as output:
    json.dump({'status': 'passed', 'files': verified}, output, indent=2)
print('Verified files:', len(verified))
PY
```

The image-import administrator must independently verify the archive SHA and selected platform/config/layers before scheduling. This transport command doesn't import an image. After the smoke, copy only the new evidence and approved metadata back with SCP/SFTP, verify the checksums locally, then exit the foreground pod or delete only its exact recorded pod name. Keep persistent inputs and failed-run evidence.
