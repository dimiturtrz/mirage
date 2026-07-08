"""Coreset subsampling for the PatchCore memory bank — pure torch kNN, no backbone dependency.

Split out from patchcore.py so the selection math imports (and tests) without torchvision: k-center
greedy picks the point farthest from the current selection each step, giving a spread-out bank.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FitCfg:
    batch: int = 16
    coreset: float = 0.1
    method: str = "greedy"
    seed: int = 0
    pool_cap: int = 200_000


@torch.no_grad()
def greedy_coreset(feats, n, seed):
    """k-center greedy: iteratively pick the point farthest from the current selection (no host
    syncs in the loop)."""
    g = torch.Generator(device=feats.device).manual_seed(seed)
    sel = torch.empty(n, dtype=torch.long, device=feats.device)
    sel[0] = torch.randint(len(feats), (1,), generator=g, device=feats.device)
    min_d = torch.cdist(feats, feats[sel[0:1]]).squeeze(1)
    for i in range(1, n):
        sel[i] = min_d.argmax()
        torch.minimum(min_d, torch.cdist(feats, feats[sel[i:i + 1]]).squeeze(1), out=min_d)
    return feats[sel]
