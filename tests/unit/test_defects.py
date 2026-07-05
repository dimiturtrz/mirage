"""Unit test for the classical defect synthesis — channel-aware, coherent defects. No dataset/GPU.

Guarantees: defect lands ON the object, ONLY masked pixels change, defect-sized region, deterministic,
and channel-aware — xyz edits the geometry (z) only, rgb edits appearance.
"""
import numpy as np
import torch

from core.data.defects import KINDS, synthesize


def _sample(b=4, c=3, h=64, w=64):
    x = torch.rand(b, c, h, w)
    valid = torch.zeros(b, 1, h, w)
    valid[:, :, 16:48, 16:48] = 1.0          # a central object region
    return x, valid


def test_defect_on_object_and_only_mask_changes():
    x, valid = _sample()
    aug, mask = synthesize(x, valid, np.random.RandomState(0), channels=("rgb",))
    assert mask.any()
    assert (mask.bool() & (valid < 0.5)).sum() == 0                  # defect stays ON the object
    changed = (aug != x).any(1, keepdim=True)
    assert (changed & ~mask.bool()).sum() == 0                       # ONLY masked pixels change
    assert (changed & mask.bool()).sum() > 0


def test_defect_sized_region_not_noise_field():
    _, mask = synthesize(*_sample(), rng=np.random.RandomState(1), channels=("rgb",))
    assert 0.0 < mask.float().mean().item() < 0.25                   # a defect, not half the frame


def test_seed_deterministic():
    x, valid = _sample()
    a1, m1 = synthesize(x, valid, np.random.RandomState(7), channels=("rgb",))
    a2, m2 = synthesize(x, valid, np.random.RandomState(7), channels=("rgb",))
    assert torch.equal(m1, m2) and torch.equal(a1, a2)


def test_xyz_edits_geometry_z_only():
    """channel-aware: on xyz, only the z channel (index 2) is displaced — x,y geometry untouched."""
    x, valid = _sample(c=3)
    aug, mask = synthesize(x, valid, np.random.RandomState(0), channels=("xyz",), kinds=("dent",))
    assert (aug[:, 0] != x[:, 0]).sum() == 0                         # x unchanged
    assert (aug[:, 1] != x[:, 1]).sum() == 0                         # y unchanged
    assert (aug[:, 2] != x[:, 2]).sum() > 0                          # z displaced (the dent)


def test_each_kind_runs():
    x, valid = _sample()
    for k in KINDS:
        _, mask = synthesize(x, valid, np.random.RandomState(0), channels=("rgb",), kinds=(k,))
        assert (mask.bool() & (valid < 0.5)).sum() == 0
