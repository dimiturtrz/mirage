"""Unit test for the channel-stacking core of the GPU split (the disk-reading GpuSplit is pragma'd).

_stack_channels turns a per-sample dict of (H,W,3) channel arrays into one (C,H,W) float32 array in
the requested channel order — the pure transform that feeds the GPU-resident tensors.
"""
import numpy as np

from core.data.static.dataset import GpuSplit


def test_stack_channels_order_and_shape():
    a = {"xyz": np.ones((4, 5, 3), np.float64), "rgb": np.zeros((4, 5, 3), np.float64)}
    out = GpuSplit._stack_channels(a, ("xyz", "rgb"))
    assert out.shape == (6, 4, 5) and out.dtype == np.float32
    assert out[:3].all() and not out[3:].any()          # xyz block first, then rgb
