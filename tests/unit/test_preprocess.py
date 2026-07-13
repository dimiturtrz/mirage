"""Unit test for the preprocess contract (docs/PLAN.md) — synthetic input, no dataset needed.

Equivalence-class on the output guarantees: fixed grid, invalid zeroed, gt within valid, the
flying-pixel cluster dropped (xyz bounded by the centroid clip).
"""
import numpy as np

from core.data.preprocess import CLIP, SIZE, Preprocess


def _sample(h=200, w=200):
    yy, xx = np.mgrid[0:h, 0:w]
    obj = (yy - 100) ** 2 + (xx - 100) ** 2 < 60 ** 2          # central object disc
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[obj] = np.stack([xx[obj] * 1e-3, yy[obj] * 1e-3, np.full(obj.sum(), 0.5)], 1)
    xyz[10:14, 10:14] = [0.0, 0.0, 5.0]                        # a far flying-pixel cluster
    rgb = (np.random.RandomState(0).rand(h, w, 3) * 255).astype(np.uint8)
    gt = np.zeros((h, w), np.uint8)
    gt[95:105, 95:105] = 255
    return rgb, xyz, gt


def test_preprocess_contract():
    out = Preprocess.preprocess(*_sample())
    assert out["xyz"].shape == (SIZE, SIZE, 3)
    assert out["rgb"].shape == (SIZE, SIZE, 3)
    assert out["valid"].dtype == bool

    invalid = ~out["valid"]
    assert np.allclose(out["xyz"][invalid], 0)                # invalid pixels zeroed
    assert np.allclose(out["rgb"][invalid], 0)
    assert not (out["gt"][invalid] > 0).any()                 # gt only within valid

    assert np.abs(out["xyz"][out["valid"]]).max() < CLIP + 0.1   # flying cluster dropped by the clip
    assert out["valid"].any()
