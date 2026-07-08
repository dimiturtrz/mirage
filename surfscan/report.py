"""The single reporting entry point — score a run, refresh numbers, render docs/cards.

    python -m surfscan.report evaluate --run-id <id>   # score a trained model back into its run
    python -m surfscan.report results                  # refresh docs/RESULTS.json from mlflow
    python -m surfscan.report sync                      # render docs/RESULTS.md tables from the json
    python -m surfscan.report modelcard                 # regenerate docs/MODEL_CARD.md
    python -m surfscan.report triad-summary             # the sim-to-real gap across seeds

Each is a `Spec` in its module; this front-end builds subparsers and routes. See also `surfscan.run`
(experiments) and `surfscan.viz` (visualization).
"""
from __future__ import annotations

from surfscan.dispatch import dispatch
from surfscan.evaluation import evaluate, modelcard, results, sync_numbers
from surfscan.experiments import triad_summary

REPORTS = [evaluate.SPEC, results.SPEC, sync_numbers.SPEC, modelcard.SPEC, triad_summary.SPEC]

if __name__ == "__main__":
    dispatch("surfscan.report", REPORTS)
