"""Unit test for the per-sample FPFH descriptor (BTF geometry half) — open3d, small synthetic cloud.

A bumpy grid surface -> per-point 33-dim FPFH + (row,col) coords. Shapes + the FpfhCfg knn knobs;
smoke-level (guards the normals -> FPFH pipeline wiring).
"""
import numpy as np

from surfscan.models.fpfh_bank import FpfhCfg, fpfh_for_sample


def _surface(g=20):
    ys, xs = np.mgrid[0:g, 0:g].astype(np.float32)
    z = 50.0 + 0.1 * np.sin(xs) * np.cos(ys)        # offset in z so normals face the origin (camera)
    xyz = np.stack([xs - g / 2, ys - g / 2, z], -1).astype(np.float32)
    return xyz, np.ones((g, g), bool)


def test_fpfh_shapes():
    xyz, valid = _surface(20)
    f, coords = fpfh_for_sample(xyz, valid, FpfhCfg(normal_knn=8, fpfh_knn=10))
    assert f.shape == (400, 33)
    assert coords.shape == (400, 2)
    assert f.dtype == np.float32


def test_fpfh_clean_subsets_points():
    xyz, valid = _surface(20)
    f, coords = fpfh_for_sample(xyz, valid, FpfhCfg(normal_knn=8, fpfh_knn=10, clean=True, graze=0.1))
    assert f.shape[1] == 33
    assert 0 < f.shape[0] <= 400 and len(coords) == f.shape[0]
