"""Recompute a run's metrics/CIs/ECE from its persisted predictions — OFFLINE, no fit/score, no GPU.

Every headline number is a pure function of the per-image predictions the harness already persisted
(`predictions.npz`). So when the metric code changes — a new bootstrap, a macro-vs-micro fix, a fresh
calibration metric — you don't re-run the pipeline: reload the predictions and recompute. This is the
command that would have turned this project's macro-CI fix from a full GPU re-run into two seconds.

    python -m surfscan.report recompute --run-id <mlflow-run-id>   # re-log updated numbers into the run
"""
from __future__ import annotations

import argparse
from typing import Any

from surfscan import tracking
from surfscan.dispatch import Spec
from surfscan.evaluation import harness


class Recompute:
    """Recompute a run's numbers from its persisted predictions.npz and re-log them into the same run."""

    @staticmethod
    def recompute(run_id: str) -> dict[str, Any]:  # pragma: no cover  mlflow IO; from_artifact is the pure core
        with tracking.Tracker.load_npz(run_id) as d:
            res = harness.Harness.from_artifact(d)
        with tracking.Tracker.resume(run_id):
            harness.Harness.log(res)           # updated metrics/brackets land back in the same run
        return res

    @staticmethod
    def args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--run-id", required=True)

    @staticmethod
    def run(args: argparse.Namespace) -> None:  # pragma: no cover  CLI glue
        Recompute.recompute(args.run_id)


SPEC = Spec("recompute", Recompute.args, Recompute.run)
