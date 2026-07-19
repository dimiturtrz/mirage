"""PatchCore-style feature memory bank — the SOTA paradigm (use the wheel, don't reinvent it).

A frozen ImageNet backbone gives semantic patch features. We store NORMAL patch features in a
memory bank (coreset-subsampled), then score each test patch by its nearest-neighbor distance to
the bank. Far from every normal patch = anomalous.

Why this beats reconstruction here: it COMPARES TO NORMAL. A normal-but-complex region (the bagel
rim) matches a normal rim in the bank -> not flagged. Reconstruction couldn't tell "hard to rebuild"
from "anomalous"; this can.

No training — just feature extraction + kNN. Run on the rgb channel (image-like).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from jaxtyping import Float
from torch import Tensor

from core.compute import Compute
from surfscan.models.coreset import Coreset, FitCfg

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PatchCore:
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        device: str = "cuda",
        *,
        amp: bool = False,
    ):
        self.device = device
        self.amp = amp                          # bf16 backbone forward (opt-in; fp32 default = reported number)
        self.layers = layers
        self._feats: dict[str, Tensor] = {}
        net = getattr(torchvision.models, backbone)(weights="DEFAULT").to(device).eval()
        for p in net.parameters():
            p.requires_grad_(requires_grad=False)
        for name in layers:
            getattr(net, name).register_forward_hook(self._save(name))
        self.net = net
        self.mean, self.std = _MEAN.to(device), _STD.to(device)
        self.bank: Tensor | None = None

    def _save(self, name: str) -> Callable[[Any, Any, Tensor], None]:
        def hook(_m: Any, _i: Any, out: Tensor) -> None:
            self._feats[name] = out
        return hook

    @torch.no_grad()
    def _embed(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """x: N,3,H,W in [0,1] -> patch features (N, h*w, C) and (h, w)."""
        x = (x - self.mean) / self.std
        self._feats = {}
        with Compute.autocast(x, amp=self.amp):
            self.net(x)
        fmaps = [self._feats[layer].float() for layer in self.layers]  # bf16 backbone -> fp32 for the bank math
        h, w = fmaps[0].shape[-2:]                                  # align to the largest (shallowest) layer
        fmaps = [F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False) for f in fmaps]
        f = torch.cat(fmaps, 1)                                     # N,C,h,w
        f = F.avg_pool2d(f, 3, stride=1, padding=1)                 # locally-aware patches (PatchCore)
        n, c = f.shape[0], f.shape[1]
        return f.permute(0, 2, 3, 1).reshape(n, h * w, c), (h, w)

    @torch.no_grad()
    def fit(self, x: Tensor, cfg: FitCfg | None = None) -> PatchCore:
        cfg = cfg or FitCfg()
        feats = []
        for i in range(0, len(x), cfg.batch):
            e, _ = self._embed(x[i:i + cfg.batch])
            feats.append(e.reshape(-1, e.shape[-1]))
        feats = torch.cat(feats)                                    # M,C
        g = torch.Generator(device=feats.device).manual_seed(cfg.seed)
        pool = feats
        if len(pool) > cfg.pool_cap:                                # cap the pool so greedy stays tractable
            pool = pool[torch.randperm(len(pool), generator=g, device=feats.device)[:cfg.pool_cap]]
        target = max(1, int(len(pool) * cfg.coreset))               # coreset of the capped pool -> bounded bank
        if cfg.method == "greedy":
            self.bank = Coreset.greedy_coreset(pool, target, cfg.seed)
        else:
            self.bank = pool[torch.randperm(len(pool), generator=g, device=pool.device)[:target]]
        return self

    @torch.no_grad()
    def score_maps(self, x: Tensor, batch: int = 8, qchunk: int = 8192) -> Float[np.ndarray, "n h w"]:
        """-> (N,H,W) anomaly maps = nearest-neighbor distance to the bank, upsampled to input size.
        Scored via bank_nn_dist (the matmul-offload form) in qchunk-row blocks so the (n*h*w, M) field
        never materializes in full — at high resolution that matrix is what spills VRAM."""
        H, W = x.shape[-2:]

        def span(i0: int, i1: int) -> Tensor:
            e, (h, w) = self._embed(x[i0:i1])                      # n,h*w,C
            n = e.shape[0]
            q = e.reshape(-1, e.shape[-1])                          # (n*h*w, C)
            nn = Compute.batched_forward(
                lambda j0, j1: Coreset.bank_nn_dist(q[j0:j1], self.bank), len(q), qchunk).reshape(n, h, w)
            m = F.interpolate(nn[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            return m.cpu()
        return Compute.batched_forward(span, len(x), batch).numpy()
