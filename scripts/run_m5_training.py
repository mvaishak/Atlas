from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from atlasrank.training.runner import run_training


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run only an approved M5 Colab pilot")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--allocation-started-utc", type=float, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recover-interrupted", action="store_true")
    args = parser.parse_args()
    report = run_training(
        input_archive=args.archive,
        checkpoint_directory=args.checkpoints,
        manifest_path=args.manifest,
        model_directory=args.model,
        input_root=args.input_root,
        persistent_directory=args.persistent,
        allocation_started_utc=args.allocation_started_utc,
        recover_interrupted=args.recover_interrupted,
        resume_checkpoint=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
