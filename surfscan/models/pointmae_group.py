"""Pure-torch farthest-point sampling + neighbourhood grouping — the CUDA-free front-end for Point-MAE.

M3DM's Point-MAE backbone groups a point cloud into local patches via two CUDA-compiled deps
(`pointnet2_ops` FPS + `knn_cuda` KNN) pinned to torch 1.9 / CUDA 11.3 — which do NOT build for
Blackwell (sm_120, RTX 5090). This module reimplements that grouping in plain torch so the geometry
backbone runs with ZERO custom CUDA ops (the #1 integration risk, avoided). Ported from the repo's own
pure-torch reference `models/pointnet2_utils.py`; see research/deep_dives/2026-07-09_m3dm-integration.md.
`farthest_point_sample`, `index_points`, `square_distance` are the primitives the op-shims in
`surfscan.models.pointmae` install for M3DM's `Group` (FPS + KNN grouping) — no CUDA ops.

Consuming the backbone weights (checked out, never vendored — CLAUDE.md):
    git clone https://github.com/nomewang/M3DM.git external/M3DM
    git -C external/M3DM checkout 62b7f468aa006f6b348f09e02f85655f416c9d76   # pinned
    # Point-MAE ShapeNet checkpoint -> external/M3DM/checkpoints/ (Google Drive, per its README)
"""
from __future__ import annotations

import torch


def square_distance(a, b):
    """Pairwise squared Euclidean distance, (B,N,3) x (B,M,3) -> (B,N,M), via the -2·a·bᵀ identity."""
    return (a.pow(2).sum(-1, keepdim=True)
            - 2 * a @ b.transpose(1, 2)
            + b.pow(2).sum(-1).unsqueeze(1))


def index_points(points, idx):
    """Gather points by index: (B,N,C) + (B,...) long -> (B,...,C), batch-aware."""
    b = points.shape[0]
    batch = torch.arange(b, device=points.device).view([b] + [1] * (idx.dim() - 1)).expand_as(idx)
    return points[batch, idx]


def farthest_point_sample(xyz, n, *, seed_first=True):
    """FPS: iteratively pick the point farthest from the current selection -> (B,n) indices. Seeding on
    point 0 (not random) keeps fit/score deterministic — the seed only shifts which equally-spread set
    is returned, and downstream features are order-invariant."""
    b, npts, _ = xyz.shape
    idx = torch.zeros(b, n, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, npts), float("inf"), device=xyz.device)
    far = torch.zeros(b, dtype=torch.long, device=xyz.device) if seed_first \
        else torch.randint(npts, (b,), device=xyz.device)
    ar = torch.arange(b, device=xyz.device)
    for i in range(n):
        idx[:, i] = far
        d = ((xyz - xyz[ar, far].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx

