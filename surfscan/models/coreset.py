"""PatchCore memory-bank construction + scoring — pure torch, no backbone dependency.

Split out from patchcore.py so the bank math imports (and tests) without torchvision. Two parts:
  - `greedy_coreset`: k-center greedy subsampling — pick the point farthest from the current selection.
  - `bank_linear` / `bank_nn_dist`: the edge-deployable scoring form. Nearest-neighbour distance to a
    fixed bank is `argmin_b ‖x−b‖² = argmin_b(‖b‖² − 2·x·b)`, so the distance field is a frozen linear
    layer (`W = −2·B`, `bias = ‖b‖²`) — a Conv1×1 that runs on an edge accelerator — followed by a
    min-reduce (the only host-side residue). See research/deep_dives/2026-07-08_patchcore_bank_on_npu.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class FitCfg:
    batch: int = 16
    coreset: float = 0.1
    method: str = "greedy"
    seed: int = 0
    pool_cap: int = 200_000


class Coreset:
    """Coreset subsampling + the edge-deployable nearest-neighbour scoring math (public names kept)."""

    @staticmethod
    @torch.no_grad()
    def greedy_coreset(feats: Float[Tensor, "m d"], n: int, seed: int) -> Float[Tensor, "n d"]:
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

    @staticmethod
    def bank_linear(bank: Float[Tensor, "m d"]) -> tuple[Tensor, Tensor]:
        """The frozen linear layer that computes the distance field `‖b‖² − 2·x·b` for a fixed bank:
        -> (weight [M,D] = −2·B, bias [M] = ‖b‖²). Load into an nn.Linear(D, M) / Conv1×1 for export —
        this is the accelerator-native half of the nearest-neighbour lookup."""
        weight = -2.0 * bank
        bias = (bank * bank).sum(1)
        return weight, bias

    @staticmethod
    @torch.no_grad()
    def bank_nn_dist(feats: Float[Tensor, "n d"], bank: Float[Tensor, "m d"]) -> Float[Tensor, "n"]:
        """Nearest-neighbour distance of each feat to the bank via the matmul form — identical to
        `torch.cdist(feats, bank).min(1).values`, but as `Linear(x) + ‖x‖²` then a min-reduce, so the
        heavy part is the accelerator-native frozen linear and only the min stays host-side."""
        weight, bias = Coreset.bank_linear(bank)
        field = feats @ weight.T + bias                        # (N,M)  ‖b‖² − 2·x·b
        xx = (feats * feats).sum(1)                            # (N,)   ‖x‖²
        d2 = xx + field.min(1).values                         # (N,)   min squared distance
        return d2.clamp_min(0).sqrt()
