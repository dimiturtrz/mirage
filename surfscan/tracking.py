"""MLflow experiment tracking — the canonical store (SQLite backend at <repo>/mlflow.db;
artifacts under <repo>/mlartifacts/).

Every method run + training run is an MLflow run; compare them with
`mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repo root. There is no separate runs/
directory — MLflow holds params, metrics (incl. per-category and per-defect), the aggregate
artifact, and trained models.

    with Tracker.run("surfscan", "patchcore", params={...}) as run_id:
        Tracker.metrics({"au_pro_mean": x})
        Tracker.per_group("au_pro", {"bagel": .., ...})
        Tracker.artifact_json("aggregate.json", agg)
        Tracker.log_model(model)              # training runs
    model = Tracker.load_model(run_id)        # later, in evaluate
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import mlflow
import mlflow.pytorch
import numpy as np
from mlflow.tracking.fluent import ActiveRun
from numpy.lib.npyio import NpzFile
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
_TRACKING_URI = f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}"

ParamScalar: TypeAlias = str | int | float | bool | None
ParamTree: TypeAlias = "ParamScalar | dict[str, ParamTree]"
JsonValue: TypeAlias = "ParamScalar | Sequence[JsonValue] | Mapping[str, JsonValue]"


@dataclass(frozen=True)
class RunParams:
    """Identity of one tracked run — the method name and the categories it covered."""

    method: str
    cats: list[str]

    def as_dict(self) -> dict[str, str]:
        return {"method": self.method, "cats": ",".join(self.cats)}


class Tracker:
    """MLflow tracking namespace — the canonical store glue (run context, metrics, artifacts, models)."""

    @staticmethod
    def _ensure_backend() -> None:  # pragma: no cover  mlflow global-state glue — repo store set lazily, not on import
        """Point mlflow at the repo SQLite store, idempotently, at first use. Lazy (not import-time) so
        importing this module never clobbers a consumer's own tracking URI."""
        if mlflow.get_tracking_uri() != _TRACKING_URI:
            mlflow.set_tracking_uri(_TRACKING_URI)

    @staticmethod
    def _flat(d: Mapping[str, ParamTree] | None, prefix: str = "") -> dict[str, ParamScalar]:
        out: dict[str, ParamScalar] = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(Tracker._flat(v, key + "."))
            else:
                out[key] = v
        return out

    @staticmethod
    @contextmanager
    def run(experiment: str, name: str, params: Mapping[str, ParamTree] | None = None,
            ) -> Iterator[str]:  # pragma: no cover  mlflow run ctx; _flat is the pure core
        Tracker._ensure_backend()
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=name) as r:
            if params:
                mlflow.log_params(Tracker._flat(params))
            yield r.info.run_id

    @staticmethod
    def resume(run_id: str) -> ActiveRun:  # pragma: no cover  mlflow run resume (network/db)
        Tracker._ensure_backend()
        return mlflow.start_run(run_id=run_id)        # context manager; log more into an existing run

    @staticmethod
    def metrics(d: Mapping[str, float | None],
                step: int | None = None) -> None:  # pragma: no cover  mlflow metric logging (network/db)
        for k, v in d.items():
            if isinstance(v, (int, float)) and not np.isnan(v):   # skip None / NaN
                mlflow.log_metric(k, float(v), step=step)

    @staticmethod
    def per_group(metric: str,
                  group_vals: Mapping[str, float | None]) -> None:  # pragma: no cover  mlflow logging (network/db)
        """Log a metric per group as <metric>/<group> (e.g. au_pro/bagel)."""
        Tracker.metrics({f"{metric}/{g}": v for g, v in group_vals.items()})

    @staticmethod
    def artifact_json(name: str, obj: JsonValue) -> None:  # pragma: no cover  mlflow artifact write (disk/db)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_text(json.dumps(obj, indent=2))
            mlflow.log_artifact(str(p))

    @staticmethod
    def artifact_npz(name: str,
                     arrays: Mapping[str, np.ndarray]) -> None:  # pragma: no cover  mlflow artifact write (disk/db)
        """Persist a flat {key: ndarray} dict as a compressed .npz run artifact (the per-image predictions)."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            # pyrefly: ignore[bad-argument-type]  savez_compressed's stub declares allow_pickle before **kwds,
            # so a Mapping[str, ndarray] splat is checked against it; every key here is an array by construction
            np.savez_compressed(p, **arrays)
            mlflow.log_artifact(str(p))

    @staticmethod
    def load_npz(run_id: str,
                 name: str = "predictions.npz") -> NpzFile:  # pragma: no cover  mlflow artifact download (network/db)
        """Download + load a run's persisted predictions -> an np.load handle (the offline-recompute input)."""
        Tracker._ensure_backend()
        p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name)
        return np.load(p, allow_pickle=False)

    @staticmethod
    def log_model(model: nn.Module) -> None:  # pragma: no cover  mlflow model serialization (disk/db)
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

    @staticmethod
    def load_model(run_id: str) -> nn.Module:  # pragma: no cover  mlflow model load (network/db)
        Tracker._ensure_backend()
        return mlflow.pytorch.load_model(f"runs:/{run_id}/model")

    @staticmethod
    def load_config(run_id: str):  # pragma: no cover  mlflow artifact download (network/db)
        """Read back the config.json artifact (the full HParams dict) logged at train time."""
        Tracker._ensure_backend()
        p = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.json")
        return json.loads(Path(p).read_text())

    @staticmethod
    def params(run_id: str) -> dict[str, str]:  # pragma: no cover  mlflow run query (network/db)
        Tracker._ensure_backend()
        return mlflow.get_run(run_id).data.params
