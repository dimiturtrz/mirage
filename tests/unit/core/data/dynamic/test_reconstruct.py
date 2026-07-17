"""Unit test for the digital-twin reconstruction — the pure organized-cloud -> valid-points filter.

_valid_points drops the sensor's misses (0/NaN entries in the organized position map) and keeps the
finite, non-origin surface points. Equivalence classes: all-valid, mixed (zeros + NaN dropped),
all-invalid -> empty.
"""
import numpy as np

from core.data.dynamic.reconstruct import CategoryReconstructor


def test_valid_points_keeps_finite_nonzero():
    xyz = np.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                    [[np.nan, 1.0, 1.0], [4.0, 5.0, 6.0]]], np.float32)   # keep 2, drop origin + NaN
    out = CategoryReconstructor._valid_points(xyz)
    assert out.shape == (2, 3)
    assert {tuple(p) for p in out} == {(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)}


def test_valid_points_all_invalid_is_empty():
    xyz = np.zeros((3, 3, 3), np.float32)
    out = CategoryReconstructor._valid_points(xyz)
    assert out.shape == (0, 3)
