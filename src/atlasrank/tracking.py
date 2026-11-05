"""A local MLflow integration check with synthetic metadata and no training."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackingEvidence:
    """Pointers to a persisted run and its verified exported artifact."""

    run_id: str
    tracking_uri: str
    artifact_uri: str
    artifact_sha256: str
    export_path: str


def tracking_smoke(root: Path) -> TrackingEvidence:
    """Write, reopen, and export a synthetic run using only local storage.

    An explicit client URI avoids changing MLflow's process-global tracking state
    or inheriting a remote MLFLOW_TRACKING_URI. Each invocation creates a new run.
    """
    from mlflow import MlflowClient

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{root / 'tracking.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "atlasrank-foundation"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=(root / "artifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id, tags={"purpose": "M1 synthetic tracking check"})
    run_id = run.info.run_id
    try:
        client.log_param(run_id, "milestone", "M1")
        client.log_metric(run_id, "fixture_value", 1.0)
        artifact = root / "inputs" / run_id / "fixture.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps({"kind": "tracking_fixture", "trained_model": False}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        client.log_artifact(run_id, str(artifact))
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise

    # Reopen through a fresh client; don't validate only an in-memory return value.
    reader = MlflowClient(tracking_uri=tracking_uri)
    persisted = reader.get_run(run_id)
    if (
        persisted.info.status != "FINISHED"
        or persisted.data.params.get("milestone") != "M1"
        or persisted.data.metrics.get("fixture_value") != 1.0
    ):
        raise RuntimeError("Local MLflow metadata round trip failed")
    export_dir = root / "exports" / run_id
    export_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(reader.download_artifacts(run_id, "fixture.json", str(export_dir)))
    if exported.read_bytes() != artifact.read_bytes():
        raise RuntimeError("Local MLflow artifact round trip failed")
    evidence = TrackingEvidence(
        run_id=run_id,
        tracking_uri=tracking_uri,
        artifact_uri=str(persisted.info.artifact_uri),
        artifact_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
        export_path=str(exported),
    )
    (export_dir / "run.json").write_text(
        json.dumps(asdict(evidence), indent=2) + "\n", encoding="utf-8"
    )
    return evidence
