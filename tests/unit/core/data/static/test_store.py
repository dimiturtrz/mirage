"""Unit tests for the store's pure metadata logic — param_key, schema, row-building.

No disk IO: _row is fed a fake Sample + numpy arrays. Equivalence classes: defective vs good sample
(label + has_gt), the derived H/W/n_valid/defect_frac.
"""
from pathlib import Path

import numpy as np
import polars as pl

from core.data.static import mvtec, store


def _sample(defect="hole"):
    gt = Path("g.png") if defect != "good" else None
    return mvtec.Sample("bagel", "test", defect, 0, Path("r.png"), Path("x.tiff"), gt)


def test_param_key_deterministic_and_varies():
    k = store.Store.param_key(256, 20, 2.0)
    assert isinstance(k, str) and "256" in k
    assert store.Store.param_key(128, 20, 2.0) != k


def test_dataset_dir_under_processed(monkeypatch, tmp_path):
    monkeypatch.setenv("SURFSCAN_DATA", str(tmp_path))
    d = store.Store.dataset_dir(256, 20, 2.0)
    assert d == tmp_path / "processed" / "mvtec3d" / store.Store.param_key(256, 20, 2.0)


def test_schema_keys_and_types():
    s = store.Store._schema()
    assert set(s) == set(store.META_FIELDS)
    assert s["label"] == pl.Int64 and s["has_gt"] == pl.Boolean and s["defect_frac"] == pl.Float64


def test_row_defective():
    valid = np.ones((4, 4), bool)
    gt = np.zeros((4, 4), np.uint8)
    gt[0, 0] = 1
    r = store.Store._row(_sample("hole"), {"valid": valid, "gt": gt}, "bagel_test_hole_000.npz")
    assert r["sample_id"] == "bagel_test_hole_000" and r["label"] == 1 and r["has_gt"] is True
    assert r["H"] == 4 and r["W"] == 4 and r["n_valid"] == 16
    assert abs(r["defect_frac"] - 1 / 16) < 1e-6


def test_row_good_has_no_gt():
    valid = np.ones((2, 2), bool)
    r = store.Store._row(_sample("good"), {"valid": valid, "gt": np.zeros((2, 2), np.uint8)}, "f.npz")
    assert r["label"] == 0 and r["has_gt"] is False and r["defect_frac"] == 0.0
