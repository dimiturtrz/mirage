"""Integration: preprocess -> dataset channel-stacking agree on the contract.

preprocess emits a per-channel (H,W,3) dict + a valid mask; the dataset layer stacks selected
channels into the (C,H,W) tensor the models consume. This pins the boundary (keys, shape, dtype,
the invalid-zeroed guarantee surviving the stack) without any dataset or GPU.
"""
import numpy as np

from core.data.dataset import _stack_channels
from core.data.preprocess import SIZE, preprocess


def _sample(h=200, w=200):
    yy, xx = np.mgrid[0:h, 0:w]
    obj = (yy - 100) ** 2 + (xx - 100) ** 2 < 60 ** 2
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[obj] = np.stack([xx[obj] * 1e-3, yy[obj] * 1e-3, np.full(obj.sum(), 0.5)], 1)
    rgb = (np.random.RandomState(0).rand(h, w, 3) * 255).astype(np.uint8)
    gt = np.zeros((h, w), np.uint8)
    gt[95:105, 95:105] = 255
    return rgb, xyz, gt


def test_preprocess_feeds_channel_stack():
    out = preprocess(*_sample())

    geo = _stack_channels(out, ["xyz"])
    assert geo.shape == (3, SIZE, SIZE) and geo.dtype == np.float32

    fused = _stack_channels(out, ["xyz", "rgb"])              # the geometry+rgb 6-channel path
    assert fused.shape == (6, SIZE, SIZE)

    invalid = ~out["valid"]                                   # background zeroed in xyz survives the stack
    assert np.allclose(geo.transpose(1, 2, 0)[invalid], 0)
