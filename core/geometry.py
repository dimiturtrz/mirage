"""Grid-native surface geometry — normals (and curvature) straight off a position map.

The position map is an *organized* point cloud: `P[u,v] = (x,y,z)` on a grid. That grid gives every
pixel its neighbourhood for free, so the surface normal is just the cross-product of the two grid
tangents (finite differences of P along the grid axes) — no KDTree, fully vectorized, GPU-resident.

This is the **keep-grid** fork (learning/2026-07-06_normals-and-curvature.md). The **drop-grid** fork —
normals on an unordered cloud via Open3D KDTree-PCA — lives in surfscan/models/fpfh_bank.py; that's the
right tool once the grid is gone. Different data shape, different tool; both compute the same quantity.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def grid_normals(xyz, valid, eps: float = 1e-8):
    """Per-pixel unit surface normals from an organized point cloud.

    xyz:   (B,3,H,W) position map (metres). valid: (B,1,H,W) object mask (1=surface).
    returns (B,3,H,W) unit normals, oriented toward the camera at the origin; zero only where the
    tangent is genuinely undefined (a pixel with no valid neighbour on either grid axis).

    Normal = normalize( t_v x t_u ), t_u = dP/du (along W), t_v = dP/dv (along H). Each tangent is a
    **central** difference where both neighbours are on-object, falling back to a **one-sided**
    difference at the object boundary and around interior sensor dropouts (the (0,0,0) holes — see
    learning/2026-07-06_structured-light-sensing-and-noise.md). Central-only would blank ~1/3 of a
    real scan (dropouts + their halos); one-sided fallback keeps normals defined wherever any
    neighbour exists, so a dent can land almost anywhere on the object.
    """
    v = valid[:, 0] > 0.5                                              # (B,H,W) bool

    def tangent(axis):
        """central where both sides valid, else one-sided; returns (tangent (B,3,H,W), has (B,H,W))."""
        a = axis
        step = (slice(None), slice(None)) + ((slice(1, None), slice(None)) if a == 2
                                             else (slice(None), slice(1, None)))
        prev = (slice(None), slice(None)) + ((slice(None, -1), slice(None)) if a == 2
                                             else (slice(None), slice(None, -1)))
        d = xyz[step] - xyz[prev]                                      # forward diff, length H-1 (or W-1)
        vpair = v[step[:1] + step[2:]] & v[prev[:1] + prev[2:]]        # both endpoints on-object
        fwd = torch.zeros_like(xyz); bwd = torch.zeros_like(xyz)
        vf = torch.zeros_like(v); vb = torch.zeros_like(v)
        if a == 2:                                                    # H axis
            fwd[:, :, :-1] = d;  vf[:, :-1] = vpair                   # fwd[i] = P[i+1]-P[i]
            bwd[:, :, 1:] = d;   vb[:, 1:] = vpair                    # bwd[i] = P[i]-P[i-1]
        else:                                                        # W axis
            fwd[:, :, :, :-1] = d;  vf[:, :, :-1] = vpair
            bwd[:, :, :, 1:] = d;   vb[:, :, 1:] = vpair
        t = torch.where(vf[:, None], fwd, 0.0) + torch.where(vb[:, None], bwd, 0.0)
        return t, (vf | vb)

    t_v, has_v = tangent(2)
    t_u, has_u = tangent(3)
    n = torch.cross(t_v, t_u, dim=1)                                  # (B,3,H,W)
    n = n / (n.norm(dim=1, keepdim=True) + eps)

    good = (v & has_v & has_u)[:, None]                              # normal defined here

    # orient toward camera (origin): normal should oppose the view ray (surface->origin = -position)
    flip = (n * xyz).sum(1, keepdim=True) > 0                         # points away from origin
    n = torch.where(flip, -n, n)
    return torch.where(good, n, torch.zeros_like(n))
