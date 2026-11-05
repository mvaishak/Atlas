"""Internal monitored worker. Public commands live in atlasrank.__main__."""

import sys
from pathlib import Path


def main() -> None:
    task, config, root, *output = sys.argv[1:]
    if task == "acquire":
        from atlasrank.data.acquire import acquire

        print(acquire(Path(config), Path(root)), flush=True)
    elif task == "prepare" and len(output) == 1:
        from atlasrank.data.prepare import prepare

        print(prepare(Path(config), Path(root), Path(output[0])), flush=True)
    elif task == "dense-calibrate" and len(output) == 1:
        from atlasrank.retrieval.calibration import calibrate

        print(calibrate(Path(config), Path(root), Path(output[0])), flush=True)
    elif task in {"bm25-build", "bm25-evaluate"} and len(output) == 1:
        from atlasrank.retrieval.jobs import build_job, evaluate_job

        operation = build_job if task == "bm25-build" else evaluate_job
        print(operation(Path(config), Path(root), Path(output[0])), flush=True)
    else:
        raise ValueError("Invalid internal worker arguments")


if __name__ == "__main__":
    main()
