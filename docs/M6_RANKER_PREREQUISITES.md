# M6 ranker prerequisite package v0.1

Execution stopped at the Linux runtime gate on 2026-10-30. The connected synthetic
boundary virtualenv lacks all 5 encoder distributions. Installing them would change
the environment; the user requested an already validated compatible runtime.

The private source package is `.local/atlasrank-m6-ranker-prerequisites-v1.zip`.
It contains source, fixed configuration, fixture tests, small manifests and these
instructions. It contains no dataset rows, weights, vectors, indexes or credentials.
Its `PACKAGE_MANIFEST.json` records every member's SHA-256 and byte length.

This is a runnable **prerequisite audit package**. A full real-data experiment runner
wasn't implemented after the stop gate. Neither command below trains a model, and
neither result authorizes training by itself.

Use an existing private checkout with the unchanged original artifacts and dependencies.
From the package directory, substitute paths for the existing checkout, its matching
Linux interpreter and fresh output directories:

```sh
/existing/atlas/.venv/bin/python scripts/check_m6_ranker_prerequisites.py \
  --root /existing/atlas \
  --bundle /existing/original-m4-bundle \
  --output /existing/storage/m6-byte-audit-new
```

The audit reads actual bytes and preserves a JSON report. Exit code 2 is intentional:
byte integrity alone leaves real-encoder/native/HTTP gates unrun. Existing output
directories are rejected. Missing inputs and mismatched bytes remain diagnostics;
the audit never repairs them or substitutes another bundle.

Only in the already compatible Linux CPU environment, run the existing synthetic gate
against that checkout's source, using another fresh output path:

```sh
cd /existing/atlas
.venv/bin/python scripts/check_xgboost_boundary.py \
  --output /existing/storage/m6-native-boundary-new
```

Don't run that command on the Mac. The original successful Linux gate and its
381 returned evidence checksums were freshly verified during this audit.

Before any fit, the full remaining Phase 1 must execute: semantic catalog/vector/native
index alignment, frozen scoring recipes, actual M4 encoder feature assembly, deterministic
real development cases, post-DMatrix offline/HTTP equality, and all resource limits.
The runtime needs verified CPU encoder libraries alongside native XGBoost and FAISS.
The full training runner must enforce the 2-hour wall, 12 GiB RSS, 8 GiB scratch,
4 GiB new persistent-output and single-fit limits. This inventory script doesn't
implement those job controls because it never executes the job.

The exact requested fit settings are in `configs/m6-ranker-fixed-v1.json`. Use complete
judged training groups, all ESCI labels and the frozen five-feature schema. Development
judgments stay out of fitting; official test judgments stay locked. Objective-update
sanity, one 100-round fit, A/C evaluation and fitted-model HTTP parity remain unrun.

The original matching local bundle is:
`data/esci/infrastructure/m4-full-local-v1/attempt-bd83a1bac3b54cdab48ca32e79f0bf90/bundle`.
All 3 component hashes match the frozen M4 CPU manifest. Preserve this directory.
The report-only return directory omits these components. No download, re-encoding or
index rebuild is needed to obtain these verified local bytes.

See [the review packet](reviews/M6.ranker.md) and
machine-readable evidence (`docs/reviews/M6.ranker.validation.json`, private evidence) for the current stop.
