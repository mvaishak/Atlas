"""Create a tiny lexical-only serving fixture or check a running fixture API."""

import argparse
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--prepare", type=Path)
    commands.add_argument("--url")
    args = parser.parse_args()
    if args.prepare:
        import polars as pl

        from atlasrank.artifacts.dense import digest
        from atlasrank.artifacts.serving_bundle import prepare
        from atlasrank.retrieval.bm25 import build

        root = args.prepare
        root.mkdir(parents=True, exist_ok=False)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        build(
            pl.DataFrame(
                {
                    "product_locale": ["us", "us"],
                    "product_id": ["a", "b"],
                    "product_title": ["red apple", "blue berry"],
                }
            ),
            artifacts / "lexical",
            {"catalog_sha256": digest("m7-smoke-fixture-v1")},
        )
        prepare(artifacts, root / "descriptor", {"lexical": "lexical"}, ["bm25"])
        print(json.dumps({"fixture": "lexical_only", "root": str(root)}))
        return
    base = args.url.rstrip("/")
    with urllib.request.urlopen(base + "/health/ready", timeout=5) as response:
        assert json.load(response)["ready"] is True
    request = urllib.request.Request(
        base + "/search",
        data=json.dumps({"query": "apple", "pipeline": "bm25"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        result = json.load(response)
    if result["pipeline_id"] != "bm25" or result["results"][0]["product_id"] != "a":
        raise ValueError("Fixture HTTP ranking differs")
    with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
        if b"atlasrank_requests_total" not in response.read():
            raise ValueError("Metrics unavailable")
    print(json.dumps({"status": "passed", "mode": "fixture", "bundle_id": result["bundle_id"]}))


if __name__ == "__main__":
    main()
