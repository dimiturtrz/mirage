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

import torch
import torch.nn.functional as F
import torchvision

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def _greedy_coreset(feats, n, seed):
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


class PatchCore:
    def __init__(self, backbone="wide_resnet50_2", layers=("layer2", "layer3"), device="cuda"):
        self.device = device
        self.layers = layers
        self._feats = {}
        net = getattr(torchvision.models, backbone)(weights="DEFAULT").to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        for name in layers:
            getattr(net, name).register_forward_hook(self._save(name))
        self.net = net
        self.mean, self.std = _MEAN.to(device), _STD.to(device)
        self.bank = None

    def _save(self, name):
        def hook(_m, _i, out):
            self._feats[name] = out
        return hook

    @torch.no_grad()
    def _embed(self, x):
        """x: N,3,H,W in [0,1] -> patch features (N, h*w, C) and (h, w)."""
        x = (x - self.mean) / self.std
        self._feats = {}
        self.net(x)
        fmaps = [self._feats[l] for l in self.layers]
        h, w = fmaps[0].shape[-2:]                                  # align to the largest (shallowest) layer
        fmaps = [F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False) for f in fmaps]
        f = torch.cat(fmaps, 1)                                     # N,C,h,w
        f = F.avg_pool2d(f, 3, stride=1, padding=1)                 # locally-aware patches (PatchCore)
        n, c = f.shape[0], f.shape[1]
        return f.permute(0, 2, 3, 1).reshape(n, h * w, c), (h, w)

    @torch.no_grad()
    def fit(self, x, batch=16, coreset=0.1, method="greedy", seed=0, pool_cap=200_000):
        feats = []
        for i in range(0, len(x), batch):
            e, _ = self._embed(x[i:i + batch])
            feats.append(e.reshape(-1, e.shape[-1]))
        feats = torch.cat(feats)                                    # M,C
        target = max(1, int(len(feats) * coreset))
        g = torch.Generator(device=feats.device).manual_seed(seed)
        if method == "greedy":
            pool = feats
            if len(pool) > pool_cap:                                # cap the pool so greedy stays tractable
                pool = pool[torch.randperm(len(pool), generator=g, device=feats.device)[:pool_cap]]
            self.bank = _greedy_coreset(pool, min(target, len(pool)), seed)
        else:
            self.bank = feats[torch.randperm(len(feats), generator=g, device=feats.device)[:target]]
        return self

    @torch.no_grad()
    def score_maps(self, x, batch=8):
        """-> (N,H,W) anomaly maps = nearest-neighbor distance to the bank, upsampled to input size."""
        H, W = x.shape[-2:]
        out = []
        for i in range(0, len(x), batch):
            e, (h, w) = self._embed(x[i:i + batch])                # n,h*w,C
            n = e.shape[0]
            d = torch.cdist(e.reshape(-1, e.shape[-1]), self.bank)  # (n*h*w, M)
            nn = d.min(1).values.reshape(n, h, w)
            m = F.interpolate(nn[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            out.append(m.cpu())
        return torch.cat(out).numpy()
