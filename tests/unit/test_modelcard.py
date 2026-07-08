"""Unit test for the model-card renderer — the pure core extracted from the mlflow/IO shell.

_render(run_name, row) builds the markdown from a metrics/params row dict (no mlflow, no file write).
Equivalence classes: a full row (headline + per-cat + params) vs an empty row (the no-metrics fallback).
"""
from surfscan.evaluation.modelcard import _render


def test_render_full_row():
    row = {
        "params.coreset": "0.1",
        "metrics.au_pro_mean": 0.908, "metrics.img_auroc_mean": 0.819,
        "metrics.au_pro/bagel": 0.928, "metrics.img_auroc/bagel": 0.943,
    }
    md = _render("patchcore_rgb_greedy", row)
    assert "# Model card — patchcore_rgb_greedy" in md
    assert "AU-PRO** 0.908" in md and "AUROC** 0.819" in md
    assert "| bagel | 0.943 | 0.928 |" in md
    assert "| coreset | 0.1 |" in md
    assert "Honest limits" in md


def test_render_no_metrics():
    md = _render("x", {})
    assert "no aggregate metrics" in md
