"""Unit tests for the RESULTS.json refresh helpers — the pure metric-extraction logic (no mlflow).

_get rounds + skips None/NaN; _apply copies present metrics into the target dict. Equivalence classes:
present value / NaN / missing key.
"""
from surfscan.evaluation.results import Results


def test_get_rounds_present():
    assert Results._get({"metrics.x": 1.23456}, "x") == 1.235


def test_get_skips_nan_and_missing():
    assert Results._get({"metrics.x": float("nan")}, "x") is None
    assert Results._get({}, "x") is None


def test_apply_sets_present_fields():
    obj = {}
    Results._apply(obj, {"metrics.au_pro_mean": 0.9, "metrics.img_auroc_mean": 0.8},
           (("au_pro_mean", "au_pro"), ("img_auroc_mean", "img_auroc")))
    assert obj == {"au_pro": 0.9, "img_auroc": 0.8}


def test_apply_skips_missing():
    obj = {}
    Results._apply(obj, {}, (("au_pro_mean", "au_pro"),))
    assert obj == {}


def test_table_mean_derives_from_rows():
    rows = [{"au_pro": 0.90, "img_auroc": 0.80}, {"au_pro": 0.80, "img_auroc": 0.60}]
    assert Results._table_mean(rows) == {"au_pro": 0.85, "img_auroc": 0.70}   # mean can't drift from its rows
