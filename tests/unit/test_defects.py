"""Unit test for the classical defect synthesis — channel-aware, coherent defects. No dataset/GPU.

Guarantees: defect lands ON the object, ONLY masked pixels change, defect-sized region, deterministic,
and channel-aware — xyz displaces geometry along the surface normal, rgb edits appearance.
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


def test_xyz_displaces_along_surface_normal():
    """channel-aware: on xyz the dent is pushed along the local surface NORMAL (not world-z). On a
    tilted plane z=a·x+b·y the normal is const (a,b,-1)/|.|, so every displacement is parallel to it —
    and x,y move too (proving it's normal-aware, not the old z-only behaviour)."""
    h = w = 64
    a, b = 0.5, 0.3
    xx = torch.arange(w).float()[None, :].expand(h, w)
    yy = torch.arange(h).float()[:, None].expand(h, w)
    zz = a * xx + b * yy                                             # a tilted plane
    x = torch.stack([xx, yy, zz])[None]                             # (1,3,H,W)
    valid = torch.zeros(1, 1, h, w); valid[:, :, 12:52, 12:52] = 1.0

    aug, mask = synthesize(x, valid, np.random.RandomState(0), channels=("xyz",), kinds=("dent",))
    d = (aug - x)[0]                                                # (3,H,W)
    moved = d.norm(dim=0) > 1e-6
    assert moved.sum() > 0

    nrm = torch.tensor([a, b, -1.0]); nrm = nrm / nrm.norm()        # the plane's constant normal
    dv = d.permute(1, 2, 0)[moved]                                  # (K,3) displacement vectors
    parallel = torch.cross(dv, nrm.expand_as(dv), dim=1).norm(dim=1) / dv.norm(dim=1)
    assert parallel.max() < 1e-4                                    # displacement ∥ surface normal
    assert (d[0].abs() > 1e-6).sum() > 0                            # x moved -> not world-z-only


def test_each_kind_runs():
    x, valid = _sample()
    for k in KINDS:
        _, mask = synthesize(x, valid, np.random.RandomState(0), channels=("rgb",), kinds=(k,))
        assert (mask.bool() & (valid < 0.5)).sum() == 0
