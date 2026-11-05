"""Full CUDA supervisor entry point; approval before input/runtime/process work."""

import argparse
from pathlib import Path

from atlasrank.retrieval.colab_full_contract import approve_full
from atlasrank.retrieval.colab_runner import execute_session


def run(
    manifest: Path, inputs: Path, persistent: Path, work: Path, allocation_started_utc: float
) -> None:
    spec = approve_full(manifest, inputs, persistent, verify_payload=False)
    execute_session(
        manifest,
        inputs,
        persistent,
        work,
        spec,
        worker_module="atlasrank.retrieval.colab_full_worker",
        allocation_started_utc=allocation_started_utc,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "inputs", "persistent", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--allocation-started-utc",
        type=float,
        required=True,
        help="UTC epoch of this runtime allocation, before uploads/setup",
    )
    args = parser.parse_args()
    run(args.manifest, args.inputs, args.persistent, args.work, args.allocation_started_utc)


if __name__ == "__main__":
    main()
