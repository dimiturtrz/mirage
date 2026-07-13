"""Unit test for tracking._flat — the pure nested-dict flattener that feeds mlflow.log_params.

Nested dicts become dotted keys (a.b.c); the mlflow calls around it are pragma'd as network/db.
"""
from surfscan.tracking import Tracker


def test_flat_nested_to_dotted():
    out = Tracker._flat({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    assert out == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_flat_empty_and_none():
    assert Tracker._flat({}) == {}
    assert Tracker._flat(None) == {}
