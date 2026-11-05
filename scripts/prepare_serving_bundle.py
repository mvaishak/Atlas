"""Seal local serving references without loading models or copying private data."""

import argparse
import json
from pathlib import Path

from atlasrank.artifacts.serving_bundle import ServingSpec, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    if set(values) != {"components", "pipelines", "query_recipe_id"}:
        parser.error("Expected components, pipelines and query_recipe_id")
    spec = ServingSpec(
        query_recipe_id=values["query_recipe_id"],
        components=values["components"],
        pipelines=values["pipelines"],
        pins={k: "0" * 64 for k in values["components"]},
    )
    prepare(args.artifacts, args.output, spec.components, spec.pipelines, spec.query_recipe_id)


if __name__ == "__main__":
    main()
