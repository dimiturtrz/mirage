"""Unit test for the classical defect synthesis — coherent defects on the object, no dataset/GPU.

Equivalence-class guarantees: the defect lands ON the object, ONLY masked pixels change (structured
edit, not global noise), the mask is a defect-sized region (not half the image), seed-deterministic.
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
    aug, mask = synthesize(x, valid, np.random.RandomState(0))
    assert mask.any()                                             # something was injected
    assert (mask.bool() & (valid < 0.5)).sum() == 0              # defect stays ON the object
    changed = (aug != x).any(1, keepdim=True)                    # per-pixel, any channel
    assert (changed & ~mask.bool()).sum() == 0                   # ONLY masked pixels change (structured)
    assert (changed & mask.bool()).sum() > 0                     # and the defect actually changed pixels


def test_defect_sized_region_not_noise_field():
    _, mask = synthesize(*_sample(), rng=np.random.RandomState(1))
    frac = mask.float().mean().item()
    assert 0.0 < frac < 0.25                                     # a defect, not half the frame


def test_seed_deterministic():
    x, valid = _sample()
    a1, m1 = synthesize(x, valid, np.random.RandomState(7))
    a2, m2 = synthesize(x, valid, np.random.RandomState(7))
    assert torch.equal(m1, m2) and torch.equal(a1, a2)


def test_each_kind_runs():
    x, valid = _sample()
    for k in KINDS:                                              # every kind produces a valid on-object mask
        _, mask = synthesize(x, valid, np.random.RandomState(0), kinds=(k,))
        assert (mask.bool() & (valid < 0.5)).sum() == 0
