"""Unit tests for the tracking carriers — `_flat` (the pure nested-dict flattener that feeds
mlflow.log_params) and `RunParams` (the identity of one tracked run).

Nested dicts become dotted keys (a.b.c); the mlflow calls around it are pragma'd as network/db.
RunParams joins one / many categories into the tracked params dict.
"""
from surfscan.tracking import RunParams, Tracker


def test_run_params_single_category():
    assert RunParams("patchcore", ["bagel"]).as_dict() == {"method": "patchcore", "cats": "bagel"}


def test_run_params_many_categories():
    assert RunParams("vae_per_category", ["bagel", "cable_gland", "carrot"]).as_dict() == {
        "method": "vae_per_category", "cats": "bagel,cable_gland,carrot"}


def test_flat_nested_to_dotted():
    out = Tracker._flat({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    assert out == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_flat_empty_and_none():
    assert Tracker._flat({}) == {}
    assert Tracker._flat(None) == {}
