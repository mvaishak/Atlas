"""Run one local CPU serving process from a sealed descriptor; never fit/download."""

import argparse
import logging
from pathlib import Path

import uvicorn

from atlasrank.serving.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--container", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        create_app(serving_bundle=args.descriptor, artifact_root=args.artifacts),
        host="0.0.0.0" if args.container else "127.0.0.1",
        port=args.port,
        workers=1,
        access_log=False,
        limit_concurrency=8,
        backlog=16,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
