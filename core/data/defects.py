"""Realistic synthetic defects on real good scans — the classical Stage-1 synthesis (channel-aware,
fully vectorized).

v1 was channel-agnostic and regressed on xyz (0.275 < Perlin 0.48): scaling xyz coordinates isn't a
dent, it's a coordinate cliff. This is **channel-aware** with **smooth profiles** (no hard edge):
  - xyz (geometry): displace the surface along its **local surface normal** by a smooth Gaussian
    profile — a real dent/bump/groove, amplitude derived from each object's own z-range (no magic
    number). Normals come off the grid (core.geometry.grid_normals), so a dent pushes into the surface,
    not down the world-z axis.
  - rgb (appearance): darken (dent/scratch), brighten (bump), or paste REAL texture (contamination) —
    never gaussian noise.
The whole batch is built with tensor ops (no per-sample python loop); the mask (profile > thr) is the
free ground-truth label. Drop-in for the DRAEM synthesize signature.

History: v1 scaled coordinates (0.275 < Perlin); v2 displaced along world-z (a dent on a tilted face
went sideways-ish); v3 (this) displaces along the true per-pixel surface normal — the physically
correct dent direction (learning/2026-07-06_normals-and-curvature.md).
"""
from __future__ import annotations

import numpy as np
import torch

from core.geometry import grid_normals

KINDS = ("dent", "scratch", "contamination", "bump")
_THR = 0.05


def _u(rng, lo, hi, b, device):
    return torch.as_tensor(rng.uniform(lo, hi, b), dtype=torch.float32, device=device)[:, None, None]


def _profiles(b, h, w, rng, device, is_scratch):
    """Batched smooth profiles (B,H,W): elliptical blob, or thin scratch where is_scratch."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    yy, xx = yy.float()[None], xx.float()[None]
    cy, cx = _u(rng, 0.2, 0.8, b, device) * h, _u(rng, 0.2, 0.8, b, device) * w
    a = _u(rng, 0, np.pi, b, device)
    ca, sa = torch.cos(a), torch.sin(a)
    y, x = yy - cy, xx - cx
    yr, xr = y * ca + x * sa, -y * sa + x * ca

    ry, rx = _u(rng, 0.03, 0.14, b, device) * h, _u(rng, 0.03, 0.14, b, device) * w
    blob = torch.clamp(1.0 - ((yr / ry) ** 2 + (xr / rx) ** 2), min=0.0)

    length, thick = _u(rng, 0.15, 0.4, b, device) * h, _u(rng, 0.004, 0.012, b, device) * h
    scratch = torch.clamp(1.0 - (xr / thick) ** 2, min=0.0) * (yr.abs() < length / 2)

    return torch.where(is_scratch[:, None, None], scratch, blob)


@torch.no_grad()
def synthesize(x, valid, rng, channels=("rgb",), kinds=KINDS):
    """x: (B,C,H,W), valid: (B,1,H,W) -> (aug, mask (B,1,H,W)). channels maps the stacked 3-wide slices:
    xyz -> geometric z-displacement, rgb -> appearance. Fully batched — no per-sample loop."""
    B, C, H, W = x.shape
    dev = x.device
    sl, off = {}, 0
    for c in channels:
        sl[c] = off; off += 3

    ki = torch.as_tensor(rng.randint(len(kinds), size=B), device=dev)
    kind = np.asarray(kinds)[ki.cpu().numpy()]
    is_scratch = torch.as_tensor(kind == "scratch", device=dev)
    is_bump = torch.as_tensor(kind == "bump", device=dev)[:, None, None]
    is_contam = torch.as_tensor(kind == "contamination", device=dev)[:, None, None, None]

    prof = _profiles(B, H, W, rng, dev, is_scratch) * (valid[:, 0] > 0.5)   # (B,H,W), on-object
    m = prof > _THR
    pa = prof * m                                                          # smooth inside, 0 outside m
    aug = x.clone()

    if "xyz" in sl:                                                        # geometry: displace along the surface normal
        xc = slice(sl["xyz"], sl["xyz"] + 3)
        xyz = aug[:, xc]                                                   # (B,3,H,W)
        n = grid_normals(xyz, valid)                                       # unit normals off the grid
        z = xyz[:, 2]
        zm = torch.where(valid[:, 0] > 0.5, z, torch.full_like(z, float("nan")))
        zr = (torch.nan_to_num(zm, nan=-1e9).amax((1, 2)) - torch.nan_to_num(zm, nan=1e9).amin((1, 2)))
        zr = zr.clamp(min=1e-6)[:, None, None]                            # object's own z-extent = scale
        sign = torch.where(is_bump, 1.0, -1.0)
        frac = torch.where(is_scratch[:, None, None], _u(rng, 0.2, 0.5, B, dev), _u(rng, 0.15, 0.35, B, dev))
        disp = (sign * frac * zr * pa)[:, None]                           # (B,1,H,W) signed magnitude
        aug[:, xc] = xyz + n * disp                                        # dent/bump normal to the surface

    if "rgb" in sl:                                                        # appearance
        r = slice(sl["rgb"], sl["rgb"] + 3)
        rgb = aug[:, r]
        s = _u(rng, 0.4, 0.7, B, dev)[:, None]                            # (B,1,1,1)
        fac = torch.where(is_bump[:, None], 1 + s * pa[:, None], 1 - s * pa[:, None])
        darkened = torch.clamp(rgb * fac, 0.0, 1.0)
        sy, sx = int(rng.uniform(0.2, 0.8) * H), int(rng.uniform(0.2, 0.8) * W)
        rolled = torch.roll(rgb, shifts=(sy, sx), dims=(2, 3))            # real texture from elsewhere
        contam = rgb * (1 - pa[:, None]) + rolled * pa[:, None]
        aug[:, r] = torch.where(is_contam, contam, darkened)

    return aug, m[:, None].to(x.dtype)
