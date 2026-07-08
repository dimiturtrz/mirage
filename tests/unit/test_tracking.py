"""Unit test for tracking._flat — the pure nested-dict flattener that feeds mlflow.log_params.

Nested dicts become dotted keys (a.b.c); the mlflow calls around it are pragma'd as network/db.
"""
from surfscan.tracking import _flat


def test_flat_nested_to_dotted():
    out = _flat({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    assert out == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_flat_empty_and_none():
    assert _flat({}) == {}
    assert _flat(None) == {}
