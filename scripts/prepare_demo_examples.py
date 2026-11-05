"""Prepare a new private five-query replay fixture; does not run search or inference."""

import argparse
import json
from pathlib import Path

from atlasrank.serving.demo_prepare import prepare_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m6-demo.json"))
    parser.add_argument("--prepared", type=Path, default=Path("data/esci/prepared-v4"))
    parser.add_argument("--lexical", type=Path, default=Path("data/esci/indexes/m3-bm25-v2"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/esci/reports/m4-cpu-eval-v1/returned/development/report.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/esci/reports/m4-cpu-eval-v1/returned/development/per-query.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_examples(
                args.config, args.prepared, args.lexical, args.report, args.records, args.output
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
