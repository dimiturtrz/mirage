"""Realistic synthetic defects on real good scans — the classical Stage-1 synthesis.

Replaces DRAEM's crude Perlin-noise blobs (which taught the discriminator to spot NOISE, not defects
— RESULTS.md 0.48). Here the anomalies are **coherent defect shapes** (dents · scratches · contamination)
with **structured, defect-like edits** (local darkening · displacement · real-texture paste — NOT
gaussian noise), placed inside the object mask. Returns `(aug, mask)`; `mask` is the free ground-truth
label. Channel-agnostic and drop-in for the DRAEM `synthesize` signature, so it also persists to a
synthetic dataset. torch-native (params via np RandomState).

The point vs Perlin: a real surface defect is a *coherent structure* (a dark dent, a scratch line, a
contamination patch), not high-frequency noise — so the detector must learn defect-ness, not noise.
"""
from __future__ import annotations

import numpy as np
import torch

KINDS = ("dent", "scratch", "contamination", "bump")


def _grid(h, w, device):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return yy.float(), xx.float()


def _blob(h, w, rng, device) -> torch.Tensor:
    """A random elliptical blob (coherent, not noise) -> bool (H,W)."""
    yy, xx = _grid(h, w, device)
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    ry, rx = rng.uniform(0.03, 0.14) * h, rng.uniform(0.03, 0.14) * w
    a = rng.uniform(0, np.pi)
    y, x = yy - cy, xx - cx
    yr = y * np.cos(a) + x * np.sin(a)
    xr = -y * np.sin(a) + x * np.cos(a)
    return (yr / ry) ** 2 + (xr / rx) ** 2 < 1.0


def _scratch(h, w, rng, device) -> torch.Tensor:
    """A thin elongated line (a scratch/crack) -> bool (H,W)."""
    yy, xx = _grid(h, w, device)
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    a = rng.uniform(0, np.pi)
    length, thick = rng.uniform(0.15, 0.4) * h, rng.uniform(0.004, 0.012) * h
    y, x = yy - cy, xx - cx
    along = y * np.cos(a) + x * np.sin(a)      # distance along the scratch
    across = -y * np.sin(a) + x * np.cos(a)    # distance across it
    return (across.abs() < thick) & (along.abs() < length / 2)


def _apply(xb, m, kind, rng):
    """Structured, defect-like edit of one sample xb (C,H,W) inside mask m (H,W bool)."""
    if kind == "dent":                                   # coherent dark depression
        xb[:, m] = xb[:, m] * rng.uniform(0.25, 0.55)
    elif kind == "scratch":                              # sharp dark line
        xb[:, m] = xb[:, m] * rng.uniform(0.15, 0.4)
    elif kind == "bump":                                 # raised / bright protrusion
        xb[:, m] = torch.clamp(xb[:, m] * rng.uniform(1.4, 1.9), max=1.0)
    elif kind == "contamination":                        # paste REAL texture from elsewhere (not noise)
        sy, sx = int(rng.uniform(0.2, 0.8) * xb.shape[1]), int(rng.uniform(0.2, 0.8) * xb.shape[2])
        rolled = torch.roll(xb, shifts=(sy, sx), dims=(1, 2))
        b = rng.uniform(0.6, 0.95)
        xb[:, m] = b * rolled[:, m] + (1 - b) * xb[:, m]


@torch.no_grad()
def synthesize(x, valid, rng, kinds=KINDS):
    """x: (B,C,H,W), valid: (B,1,H,W) bool-ish -> (aug, mask (B,1,H,W)) with a coherent defect per sample."""
    B, C, H, W = x.shape
    aug = x.clone()
    mask = torch.zeros((B, 1, H, W), dtype=x.dtype, device=x.device)
    for b in range(B):
        kind = kinds[rng.randint(len(kinds))]
        shape = _scratch(H, W, rng, x.device) if kind == "scratch" else _blob(H, W, rng, x.device)
        m = shape & (valid[b, 0] > 0.5)                  # keep the defect on the object
        if not m.any():
            continue
        _apply(aug[b], m, kind, rng)
        mask[b, 0][m] = 1.0
    return aug, mask
