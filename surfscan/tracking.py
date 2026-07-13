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
_TRACKING_URI = f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}"


class Tracker:
    """MLflow tracking namespace — the canonical store glue (run context, metrics, artifacts, models)."""

    @staticmethod
    def _ensure_backend():  # pragma: no cover  mlflow global-state glue — set the repo store lazily, not on import
        """Point mlflow at the repo SQLite store, idempotently, at first use. Lazy (not import-time) so
        importing this module never clobbers a consumer's own tracking URI."""
        if mlflow.get_tracking_uri() != _TRACKING_URI:
            mlflow.set_tracking_uri(_TRACKING_URI)

    @staticmethod
    def _flat(d, prefix=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(Tracker._flat(v, key + "."))
            else:
                out[key] = v
        return out

    @staticmethod
    @contextmanager
    def run(experiment, name, params=None):  # pragma: no cover  mlflow run context (network/db); _flat is the pure core
        Tracker._ensure_backend()
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=name) as r:
            if params:
                mlflow.log_params(Tracker._flat(params))
            yield r.info.run_id

    @staticmethod
    def resume(run_id):  # pragma: no cover  mlflow run resume (network/db)
        Tracker._ensure_backend()
        return mlflow.start_run(run_id=run_id)        # context manager; log more into an existing run

    @staticmethod
    def metrics(d, step=None):  # pragma: no cover  mlflow metric logging (network/db)
        for k, v in d.items():
            if isinstance(v, (int, float)) and v == v:   # skip None / NaN
                mlflow.log_metric(k, float(v), step=step)

    @staticmethod
    def per_group(metric, group_vals):  # pragma: no cover  mlflow metric logging (network/db)
        """Log a metric per group as <metric>/<group> (e.g. au_pro/bagel)."""
        Tracker.metrics({f"{metric}/{g}": v for g, v in group_vals.items()})

    @staticmethod
    def artifact_json(name, obj):  # pragma: no cover  mlflow artifact write (disk/db)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_text(json.dumps(obj, indent=2))
            mlflow.log_artifact(str(p))

    @staticmethod
    def log_model(model):  # pragma: no cover  mlflow model serialization (disk/db)
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

    @staticmethod
    def load_model(run_id):  # pragma: no cover  mlflow model load (network/db)
        Tracker._ensure_backend()
        return mlflow.pytorch.load_model(f"runs:/{run_id}/model")

    @staticmethod
    def load_config(run_id):  # pragma: no cover  mlflow artifact download (network/db)
        """Read back the config.json artifact (the full HParams dict) logged at train time."""
        Tracker._ensure_backend()
        p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.json")
        return json.loads(Path(p).read_text())

    @staticmethod
    def params(run_id):  # pragma: no cover  mlflow run query (network/db)
        Tracker._ensure_backend()
        return mlflow.get_run(run_id).data.params
