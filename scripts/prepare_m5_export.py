from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlasrank.training.export import build_training_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a minimal training-only M5 export")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".local/colab/m5-training-input-v6.zip"),
    )
    args = parser.parse_args()
    result = build_training_export(args.repository.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
