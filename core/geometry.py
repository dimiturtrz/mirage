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
    returns (B,3,H,W) unit normals, oriented toward the camera at the origin; zero where undefined
    (background or where a finite-difference neighbour falls off the object).

    Normal = normalize( t_v x t_u ), t_u = dP/du (along W), t_v = dP/dv (along H), via central
    differences. A pixel whose +/-neighbour along either axis is invalid gets a zero normal (the
    tangent would cross the object boundary and be garbage) — callers treat zero as "no displacement".
    """
    v = valid[:, 0] > 0.5                                              # (B,H,W) bool
    # central differences along H (v-axis) and W (u-axis)
    t_v = torch.zeros_like(xyz)
    t_u = torch.zeros_like(xyz)
    t_v[:, :, 1:-1, :] = xyz[:, :, 2:, :] - xyz[:, :, :-2, :]
    t_u[:, :, :, 1:-1] = xyz[:, :, :, 2:] - xyz[:, :, :, :-2]

    n = torch.cross(t_v, t_u, dim=1)                                   # (B,3,H,W)
    n = n / (n.norm(dim=1, keepdim=True) + eps)

    # a diff is only trustworthy if both +/- neighbours (and the pixel) are on-object
    vh = torch.zeros_like(v); vw = torch.zeros_like(v)
    vh[:, 1:-1, :] = v[:, 2:, :] & v[:, :-2, :]
    vw[:, :, 1:-1] = v[:, :, 2:] & v[:, :, :-2]
    good = (v & vh & vw)[:, None]                                      # (B,1,H,W)

    # orient toward camera (origin): normal should oppose the view ray (surface->origin = -position)
    flip = (n * xyz).sum(1, keepdim=True) > 0                          # points away from origin
    n = torch.where(flip, -n, n)
    return torch.where(good, n, torch.zeros_like(n))
