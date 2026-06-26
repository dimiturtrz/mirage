"""Optional MLflow experiment tracking — local file backend, runs/ stays canonical.

A thin, GUARDED layer: if mlflow is absent or MIRAGE_NO_MLFLOW is set, every call is a no-op,
so training/eval never depend on it. It's the cross-run comparison UI (`mlflow ui`), not the
source of truth — runs/<name>/{config,results}.json remain authoritative. Mirrors systole's
cardioseg/tracking.py.

    trk = start("mirage", run_name, params=hp.model_dump(), run_dir=out)
    trk.metric("recon", rl, step=ep); trk.end()
    trk = resume("mirage", run_name, run_dir=run); trk.metric("test_au_pro", x); trk.end()
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MLRUNS = _ROOT / "mlruns"


def _mlflow():
    if os.environ.get("MIRAGE_NO_MLFLOW"):
        return None
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


class _Noop:
    def metric(self, *a, **k): pass
    def artifact(self, *a, **k): pass
    def end(self): pass


class _Live:
    def __init__(self, mlflow):
        self._m = mlflow

    def metric(self, key, value, step=None):
        try:
            self._m.log_metric(key, float(value), step=step)
        except Exception:
            pass

    def artifact(self, path):
        try:
            if Path(path).exists():
                self._m.log_artifact(str(path))
        except Exception:
            pass

    def end(self):
        try:
            self._m.end_run()
        except Exception:
            pass


def _begin(experiment, run_name, params, run_dir, resume):
    mlflow = _mlflow()
    if mlflow is None:
        return _Noop()
    try:
        _MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(_MLRUNS.as_uri())
        mlflow.set_experiment(experiment)
        idf = Path(run_dir) / ".mlflow_run_id" if run_dir else None
        if resume and idf and idf.exists():
            mlflow.start_run(run_id=idf.read_text().strip())   # resume, don't re-log params
        else:
            mlflow.start_run(run_name=run_name)
            if params:
                mlflow.log_params(_flat(params))
            if idf:
                idf.write_text(mlflow.active_run().info.run_id)
        return _Live(mlflow)
    except Exception:
        return _Noop()


def start(experiment, run_name, params=None, run_dir=None):
    """Begin a fresh tracked run (local mlruns/)."""
    return _begin(experiment, run_name, params, run_dir, resume=False)


def resume(experiment, run_name, run_dir=None):
    """Resume the run tied to run_dir (via .mlflow_run_id) if it exists, else start fresh."""
    return _begin(experiment, run_name, None, run_dir, resume=True)
