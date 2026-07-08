"""MLflow experiment tracking — the canonical store (SQLite backend at <repo>/mlflow.db;
artifacts under <repo>/mlartifacts/).

Every method run + training run is an MLflow run; compare them with
`mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repo root. There is no separate runs/
directory — MLflow holds params, metrics (incl. per-category and per-defect), the aggregate
artifact, and trained models.

    with track.run("surfscan", "patchcore", params={...}) as run_id:
        track.metrics({"au_pro_mean": x})
        track.per_group("au_pro", {"bagel": .., ...})
        track.artifact_json("aggregate.json", agg)
        track.log_model(model)              # training runs
    model = track.load_model(run_id)        # later, in evaluate
"""
from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mlflow
import mlflow.pytorch

_ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}")


def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


@contextmanager
def run(experiment, name, params=None):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as r:
        if params:
            mlflow.log_params(_flat(params))
        yield r.info.run_id


def resume(run_id):
    return mlflow.start_run(run_id=run_id)        # context manager; log more into an existing run


def metrics(d, step=None):
    for k, v in d.items():
        if isinstance(v, (int, float)) and v == v:   # skip None / NaN
            mlflow.log_metric(k, float(v), step=step)


def per_group(metric, group_vals):
    """Log a metric per group as <metric>/<group> (e.g. au_pro/bagel)."""
    metrics({f"{metric}/{g}": v for g, v in group_vals.items()})


def artifact_json(name, obj):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        p.write_text(json.dumps(obj, indent=2))
        mlflow.log_artifact(str(p))


def log_model(model):
    mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")


def load_model(run_id):
    return mlflow.pytorch.load_model(f"runs:/{run_id}/model")


def load_config(run_id):
    """Read back the config.json artifact (the full HParams dict) logged at train time."""
    p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.json")
    return json.loads(Path(p).read_text())


def params(run_id):
    return mlflow.get_run(run_id).data.params
