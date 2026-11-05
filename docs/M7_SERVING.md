# M7 local serving

The real-model HTTP gate passed on DSMLP on 2026-11-04 Pacific. The returned evidence verifies three 100-candidate search cases and three seven-candidate rerank cases, with exact features/predictions, 58.17-second supervised wall time, 5.52 GiB sampled peak RSS and clean shutdown. See [the M7 review](reviews/M7.md) for measured evidence and limitations. That run used an isolated CPU environment with the explicit platform-string compatibility policy described in [M7_DSMLP.md](M7_DSMLP.md). Default serving startup retains exact recipe identity checks. The user accepted M7 on 2026-11-05.

M7 reuses one FastAPI process and the frozen scoring modules. BM25 stays the default. `dense_m4`, `rrf_m4` and experimental `m6_xgb_m4` require a complete validated Linux CPU runtime. Test judgments never enter serving.

## Prepare and run

The descriptor seals component reference paths and manifest pins. It doesn't copy data, load a model or certify native compatibility. Every referenced loader validates the actual component bytes before readiness. The ranker and every neural pipeline also require the separately pinned CPU query-recipe digest.

```sh
.venv/bin/python scripts/prepare_serving_bundle.py \
  --config configs/m7-serving-v1.json --artifacts . \
  --output .local/m7-serving-descriptor-new

# Compatible Linux CPU only, under an external memory/time limit:
.venv/bin/python scripts/serve_bundle.py \
  --descriptor .local/m7-serving-descriptor-new --artifacts . --port 8000
```

Use fresh descriptor paths. The current reference component locations are in [directory.md](directory.md). Keep original component directories immutable. The descriptor's paths relocate together beneath `--artifacts`. Symlinks and escaping paths fail validation.

Full startup requires 8 GiB available RAM and rejects a sampled RSS peak above 12 GiB. These checks supplement an external container/process memory limit. They can't interrupt every native allocation. The tiny lexical-only fixture uses a 256 MiB minimum. Model loading happens once at startup. Each complete-runtime request acquires the same nonblocking admission lock. Overlap returns 503 with `Retry-After: 1`. Resource admission failure clears readiness until restart.

The recovered model's CPU query recipe includes runtime versions and platform metadata. Default loading requires exact identity. The separately authorized DSMLP runner allows only the Linux kernel/glibc platform string to differ and records both recipe identities. Every other recipe field and the offline/HTTP feature and prediction comparisons remain exact. See [the DSMLP record](M7_DSMLP.md) for that explicit policy.

## API

```sh
curl -sS http://127.0.0.1:8000/health/ready
curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"wireless headphones","pipeline":"rrf_m4","top_k":10}'
curl -sS http://127.0.0.1:8000/metrics
```

`/search` preserves the existing 512-character, 128-analyzer-token and top_k 1–100 limits. Only US locale is accepted. Bodies above 16 KiB are rejected before JSON parsing, including chunked bodies. Invalid JSON and validation errors return a payload-free 422 response.

Results include the selected pipeline, serving/component versions, score type, candidate counts, ranks, request ID and observed stage timings. XGBoost scores are margins. M6 remains experimental. Complete-runtime fixture execution is labeled `deterministic_fixture`. Saved `/comparison` responses retain `saved_development`.

`/health/live` checks the process. Complete-bundle startup failure keeps `/health/ready` and configured inference unavailable. The legacy BM25-only app retains its original optional `/rerank` behavior. An unknown or disabled pipeline never falls back silently.

Metrics use bounded route/status and pipeline/stage labels. Requests and errors can be counted by status. Duration/stage sums and counts support aggregate means. M8's measured latency percentiles come from a separate fixed-workload experiment. Logs contain request IDs, bounded route names, statuses, durations and bundle identity. Exception payloads, query text and product IDs are excluded.

## CPU image and fixture smoke

The repository cleanup checked the current runtime source in image `sha256:61a1e204e009245333bdc3a3f2de99a29e63f82ee4a3460225416d93885211a1`. Its lexical fixture reached readiness after 10.24 seconds, passed search/metrics and the scheduled Docker health check, and exited with code 0. The fixture container was removed. The following original image record is retained separately.

The image ID and measurements in this section describe the earlier executed image. A rebuild from current source produces a separate image identity and needs its own fixture check. The actual DSMLP model gate used the separately pinned isolated environment in `environments/dsmlp-cpu/requirements.txt`.

Docker build and lexical fixture smoke **passed** on Linux/amd64 emulated by the local aarch64 Docker VM. Both base images are pinned by their observed registry digests. The resulting image ID is `sha256:6cebdd22ab0c98d452771d93cbff8c1a4ba8e205ae134d7e8ba0d0242712cfb7`. The build context is allowlisted and contains no private data or model artifacts. No remote image push is part of this workflow.

```sh
docker build --platform linux/amd64 -f docker/Dockerfile.cpu -t atlasrank:m7-cpu .
.venv/bin/python scripts/m7_fixture_smoke.py --prepare .local/m7-smoke-new

docker run --rm --name atlasrank-m7-fixture --platform linux/amd64 \
  --cpus=2 --memory=2g --memory-swap=2g --pids-limit=128 \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/.local/m7-smoke-new:/bundles:ro" atlasrank:m7-cpu

# In a second terminal, after readiness:
.venv/bin/python scripts/m7_fixture_smoke.py --url http://127.0.0.1:8000
```

Stop the named fixture container after the smoke check. A real neural startup needs its separately mounted complete component directories, descriptor and approved CPU environment, at least 8 GiB measured available memory and a maximum 12 GiB worker RSS. Record container image ID, source/descriptor hashes, architecture, startup/readiness time, sampled peak RSS, responses and shutdown. Cap that smoke at 300 seconds. Don't change the catalog, bundle, thread count or query recipe to squeeze through a resource failure.

A successful lexical fixture container doesn't satisfy actual-model parity or full-bundle smoke. Docker images run as UID 10001. Mounted inputs must be readable by that user. The image runs one worker, limits concurrent connections to 8 and uses no response cache. Native Mac Uvicorn fixture smoke is recorded separately from Docker evidence.

## Verification

Run fixture groups in separate processes as specified in the [M3–M5 reproduction guide](REPRODUCING_M3_M5.md). Never combine Mac Torch and FAISS tests in one process.

```sh
.venv/bin/python -m pytest -q tests/test_m7_bundle.py tests/test_m7_api.py \
  tests/test_ranking_features.py tests/test_ranking_parity.py \
  tests/test_search_api.py tests/test_demo.py tests/test_xgboost_colab_package.py
.venv/bin/ruff check src tests scripts
.venv/bin/mypy
```

The runner admission tests also need a fresh process: importing scikit-learn in another fixture sets `KMP_DUPLICATE_LIB_OK` through that dependency, which changes the runner's expected rejection reason. Leave the production rejection guard intact.

The [M7 review](reviews/M7.md) records actual checks, findings and pending gates. No M8 load benchmark or M9 optimization variant ran in M7.


## Historical container checks, 2026-11-04

The lexical fixture became ready after 11.769 seconds including Docker launch, served search/metrics, exited with code 0 and was removed. One observed memory snapshot was 146.3 MiB under a 2 GiB container limit. This isn't a peak measurement or benchmark. The container was stopped before separately observing the scheduled image HEALTHCHECK. HTTP readiness was checked directly. Other existing containers were left untouched.

The final image also imported CPU Torch 2.14.1+cpu, FAISS 1.15.1 and XGBoost 3.1.1 together, and loaded the original sealed M6 model with 100 rounds / 5 features. No prediction, feature construction, model fitting or catalog loading occurred in that check. This doesn't establish actual encoder/index serving parity.

Docker reports 8,217,448,448 total VM RAM bytes (7.65 GiB), below the 8 GiB available-memory neural admission requirement. No VM resize or relaxed gate was attempted. Full neural startup needs a larger compatible Linux runtime.

## DSMLP acceptance

The actual full neural CPU startup and HTTP gate passed in the separately approved DSMLP environment. See [the DSMLP execution record](M7_DSMLP.md) for exact measurements, provenance and historical preparation. The original local Docker VM remains below the full-model admission threshold. The user accepted M7 on 2026-11-05.

## Public source and private inputs

Fixture preparation and tests use generated tiny inputs. Full real-model reproduction requires the immutable private component tree, descriptor, reconstructed recipe/provenance and an approved execution manifest described in [directory.md](directory.md). Generated validation JSON and raw return files remain private. The source checkout records procedures and measured results, with no bundled model, catalog or historical evidence archive.
