"""Local foundation, data contracts, and the M3 lexical vertical slice."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("tracking-smoke", help="Verify local SQLite MLflow and artifacts")
    smoke.add_argument("--root", type=Path, default=Path(".local/mlflow"))
    for name in ("data-acquire", "data-prepare"):
        data = commands.add_parser(name, help="Pinned local ESCI data contract")
        data.add_argument("--config", type=Path, default=Path("configs/esci.json"))
        data.add_argument("--root", type=Path, default=Path("data/esci"))
        if name == "data-prepare":
            data.add_argument("--output", type=Path, required=True)
    for name in ("bm25-build", "bm25-evaluate", "dense-calibrate"):
        lexical = commands.add_parser(
            name, help="Bounded local lexical job or approved CPU encoder calibration"
        )
        lexical.add_argument(
            "--config",
            type=Path,
            default=Path(
                "configs/dense.json" if name == "dense-calibrate" else "configs/bm25.json"
            ),
        )
        lexical.add_argument("--root", type=Path, default=Path("data/esci"))
        lexical.add_argument("--output", type=Path, required=True)
    search = commands.add_parser("bm25-search", help="Search a validated local lexical bundle")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    serve = commands.add_parser("serve", help="Run the minimal M3 API on loopback only")
    serve.add_argument("--index", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--comparison", type=Path)
    serve.add_argument("--comparison-sha256")
    full = commands.add_parser(
        "dense-run", help="Resource-gated CPU M4 run; proposed manifests fail closed"
    )
    full.add_argument("--execution-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "dense-run":
        from atlasrank.retrieval.full_run import supervise

        print(supervise(args.execution_manifest))
        return
    if args.command in {"bm25-build", "bm25-evaluate", "dense-calibrate"}:
        from atlasrank.data.runner import run_bounded

        run_bounded(args.command, args.config, args.root, args.output)
        return
    if args.command == "bm25-search":
        from atlasrank.retrieval.bm25 import load

        print(json.dumps([asdict(hit) for hit in load(args.index).search(args.query, args.top_k)]))
        return
    if args.command == "serve":
        import logging

        import uvicorn

        from atlasrank.serving.app import create_app

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        if bool(args.comparison) != bool(args.comparison_sha256):
            parser.error("--comparison and --comparison-sha256 must be supplied together")
        uvicorn.run(
            create_app(args.index, args.comparison, args.comparison_sha256),
            host="127.0.0.1",
            port=args.port,
            access_log=False,
        )
        return
    if args.command.startswith("data-"):
        from atlasrank.data.runner import run_bounded

        run_bounded(
            args.command.removeprefix("data-"),
            args.config,
            args.root,
            getattr(args, "output", None),
        )
        return
    from atlasrank.tracking import tracking_smoke

    evidence = tracking_smoke(args.root)
    print(json.dumps(asdict(evidence), indent=2))


if __name__ == "__main__":
    main()
