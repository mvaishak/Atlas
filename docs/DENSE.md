> Current 2026-10-29 status: full-catalog dense artifacts and development-only dense/hybrid evaluation completed on Colab; see [M4.cpu-run.md](reviews/M4.cpu-run.md). Earlier execution proposals below are historical. The Mac OpenMP issue remains unresolved; M5 is not authorized.

# M4 dense infrastructure contract

The approved infrastructure is implemented within `retrieval/` and `artifacts/`; full-catalog execution remains gated. No dataset/metric/split/gain changes were made. The separate [Colab integration](COLAB.md) preserves this CPU reference and adds receipt-backed transport plus explicit CUDA-product/CPU-query provenance. GPU pilot execution awaits approval. See [M4 review](reviews/M4.md), [resource plan](M4_PLAN.md), and the optional [accelerator plan](ACCELERATORS.md).

## Canonical input and recipe

The approved M2 catalog isn't assumed to be in index order. `canonical_catalog` creates a new title/locale/product-ID projection sorted by `(product_locale, product_id)`, without modifying the source. Duplicate/empty/non-US identities, missing text normalization, nulls and wrong types fail. This one-time sort materializes title/identity columns, not embeddings; its full-catalog peak memory remains unmeasured and is covered by the future worker guard.

`PinnedEncoder.recipe` binds model/tokenizer file hashes, revision, dimensions, batch size, preprocessing, padding/pooling, dtype/backend, seed/threads and encoder implementation hash. Changing the recipe, source bytes, origin, chunk boundaries or wall budget rejects resume. The calibrated CPU recipe remains title-only, float32, batch 32 and 256 tokens. In-flight recipe changes fail before publishing a chunk.

## Immutable chunks and interruption

An embedding job contains `JOB.json`, `chunks/`, `partials/`, `attempts/`, `.writer.lock`, and, once complete, `JOB_COMPLETE.json`. Each completed chunk contains exactly:

```text
products.parquet   canonical identities and original titles for this row interval
vectors.npy        finite float32 unit vectors in exactly that order
manifest.json      schema, job identity, start/stop, file hashes and artifact identity
COMPLETE.json      manifest checksum; written last
```

Chunks align to fixed encoding batches. Work is staged in a unique partial directory, flushed/fsynced and validated, then atomically renamed under a nonblocking advisory writer lock. Completed files are read-only and application code never replaces them. Checksums and source-row comparisons protect accidental corruption/incompatibility; this isn't a signed-artifact or hostile-upload boundary.

Resume verifies existing chunks against expected source rows/titles, recipe and row bounds before skipping them. Committed chunks aren't re-encoded or rewritten. Incomplete partial directories remain preserved and aren't reused; a replacement chunk is built in a new partial directory. Global canonical order, contiguous ranges and the exact completed-chunk inventory are checked before job completion. Gaps/overlaps, altered identities or corrupt committed chunks fail closed.

Immutable attempt start/end records accumulate active command time across interruptions/restarts, including failed attempts. On recovery of an unclosed process attempt, elapsed wall time through recovery is charged conservatively, including downtime, and recorded once. Clock reversal charges the reserved remainder. That conservative charge can exhaust a budget after a long outage; it never resets the ledger. Embedding, bundle and development-evaluation phases share the job ledger.

## Dense bundle

`build_bundle` accepts only a completed chunk job and creates a fresh destination. It streams chunks into a memory-mapped float32 matrix and a CPU `IndexFlatIP`, writes the identical product mapping, and seals this exact inventory:

```text
products.parquet   full canonical identity/title mapping
vectors.npy        full ordered normalized vectors
index.faiss        exact inner-product native index
manifest.json      schema, recipe/source/chunk lineage, versions, dimensions/counts, hashes
COMPLETE.json      manifest checksum
```

Loading verifies the entire inventory and checksums before native FAISS parsing. It rejects symlinks, numerical-version/recipe/origin incompatibility, wrong index type/metric/count/dimension/dtype, nonfinite/non-unit vectors and invalid identities. Every reconstructed FAISS row is compared **exactly** to the corresponding saved float32 vector in bounded blocks. Product keys stay canonical; the evaluator additionally requires exact lexical/dense keys and titles from the same pinned source. No vector-to-product identity is inferred from an unrelated file order.

Search retains the predeclared threshold/range treatment of cutoff ties. The same native inner-product primitive scores all supplied Track A candidates without retrieval truncation. Full native bundles are trusted local artifacts; checksum validation doesn't make arbitrary foreign FAISS input a security boundary. The original M3 bundle/scorer/API remain unchanged.

## Development evaluator

`evaluate_development` accepts a preparation, lexical bundle, dense bundle, encoder recipe, exact source pins, output and cumulative job ledger. It exposes **no split selector**: only the guarded development loader is called. Source hashes, recipe, job lineage, lexical/dense keys and titles are checked before labels are opened. Test references/raw examples aren't read. Full execution is separate from implementation.

Track A scores every judged product using full-catalog BM25 and native dense cosine, then RRF over those same complete judged-pool rankings. Track B independently searches the full catalog for up to 100 BM25 positive matches and 100 dense neighbors, then takes the top 100 from their deduplicated RRF-60 union. Ranks start at 1; absent retriever results contribute 0. No judged positives are inserted into retrieval. Unknown labels remain unknown.

Existing metrics, zero-IDCG/no-E handling, query slices and 2,000-resample/seed-42 normalized-group paired intervals are reused without modification. Reports include returned counts, eligibility/exclusion counts, per-query rankings/scores and source/bundle/recipe hashes. Paired comparisons are dense minus BM25, hybrid minus BM25 and hybrid minus dense. C is marked unrun; B's saved hybrid lists provide the future pre-ranker reference. Fixture results aren't ESCI quality evidence.

The full worker also requires exact per-query BM25 metrics, ordered identities and scores against the pinned M3 development output, excluding timers. A fixture confirms that changed scores fail this check. The real full comparison hasn't run.

## Execution gate and bounded evidence

The implemented `dense-run --execution-manifest PATH` command checks explicit approval, reviewed source/lock/config hashes and fixed resource limits **before any model/data/process work**. The checked-in `configs/m4-full-run.proposed.json` is deliberately unapproved and must fail closed. After lead approval, create a separate ignored approved manifest with identical reviewed settings, not a silently modified proposal. Resume uses the same run directory and immutable manifest.

The full-run supervisor caps total active command wall time at 12 hours across attempts, samples worker RSS at 8 GiB, requires 6 GiB available RAM at launch, stops below 1 GiB during work, and enforces an 8 GiB new-run directory cap plus the existing 30 GiB data cap / 2 GiB free disk headroom. A worker lock prevents duplicate runs while a previous worker remains alive. Cooperative checks at encoding batches, bundle chunks and evaluation queries stop a worker that loses its supervisor, exceeds the inherited deadline or hits resource guards. A blocked native call may delay that checkpoint; an active orphan still requires operational handling before resume. These are cooperative/sampled guards, not OS allocation guarantees. This full worker has not been launched.

Completed bundle/report selection uses immutable pointers; incomplete outputs and prior attempts are preserved. If a failure fills the 8 GiB reservation, stop for a budget review rather than delete history. Full-build memory, checkpoint throughput and A/B quality remain unmeasured.

The following bounded check already ran with a fresh output path:

```sh
uv run --locked python scripts/check_dense_infrastructure.py \
  --pilot data/esci/calibrations/m4-cpu-v1 \
  --output data/esci/infrastructure/m4-v2 --process-interrupt
```

It replays 64 previously encoded pilot vectors, signal-terminates its own child during the second chunk, recovers an unclosed attempt, resumes and compares against an uninterrupted build. All five bundle files are byte-identical and existing chunk hashes/mtimes are unchanged. Its development fixtures use explicitly synthetic labels; they neither consume real relevance judgments nor establish catalog quality. Repetitions need fresh destinations and execution scope. Routine tests use only synthetic encoders/data and no model download.
