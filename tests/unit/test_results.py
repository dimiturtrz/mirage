"""Unit tests for the RESULTS.json refresh helpers — the pure metric-extraction logic (no mlflow).

_get rounds + skips None/NaN; _apply copies present metrics into the target dict. Equivalence classes:
present value / NaN / missing key.
"""
from surfscan.evaluation.results import _apply, _get


def test_get_rounds_present():
    assert _get({"metrics.x": 1.23456}, "x") == 1.235


def test_get_skips_nan_and_missing():
    assert _get({"metrics.x": float("nan")}, "x") is None
    assert _get({}, "x") is None


def test_apply_sets_present_fields():
    obj = {}
    _apply(obj, {"metrics.au_pro_mean": 0.9, "metrics.img_auroc_mean": 0.8},
           (("au_pro_mean", "au_pro"), ("img_auroc_mean", "img_auroc")))
    assert obj == {"au_pro": 0.9, "img_auroc": 0.8}


def test_apply_skips_missing():
    obj = {}
    _apply(obj, {}, (("au_pro_mean", "au_pro"),))
    assert obj == {}
