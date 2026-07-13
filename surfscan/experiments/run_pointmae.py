"""Point-MAE learned-geometry memory bank — the iwq lite-path (M3DM's backbone, our bank/harness).

Learned point-transformer features (Point-MAE, ShapeNet-pretrained) replace hand-FPFH as the geometry
descriptor, dropped into the PatchCore bank we already trust (greedy coreset + NN-to-normal). Skips
M3DM's UFF fusion + 3-bank machinery; runs CUDA-free (see surfscan.models.pointmae). The measured test
of "learned geometry > hand-FPFH": compare this to BTF (FPFH, AU-PRO 0.653).

Run:  python -m surfscan.run pointmae [--cats bagel ...] [--coreset 0.1] [--n-points 16384]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from core.data import store
from surfscan.evaluation import scoring
from surfscan.method_cli import method_spec
from surfscan.models.coreset import Coreset
from surfscan.models.pointmae import Pointmae
from surfscan.models.pointmae_group import PointmaeGroup


@dataclass(frozen=True)
class PointMAECfg:
    coreset: float = 0.1
    n_points: int = field(default=16384, metadata={"help": "subsample cap of valid points per cloud"})
    size: int | None = field(default=None, metadata={"help": "store resolution (default 256)"})


class PointMAEMethod:
    run_name = "pointmae_geom"

    def __init__(self, cfg: PointMAECfg, dev):
        self.cfg = cfg
        self.dev = dev
        self.size = cfg.size or 256
        self.net = Pointmae.load_pointmae(dev, ckpt="pointmae_pretrain.pth")

    def _norm(self, pts):
        """Unit-sphere normalize (ShapeNet pc_normalize) — the frame Point-MAE was pretrained on:
        centre, then scale by the max distance to the centre. -> (normalized pts, mean, scale)."""
        mean = pts.mean(0)
        scale = (pts - mean).norm(dim=1).max() + 1e-6
        return (pts - mean) / scale, mean, scale

    def _features(self, xyz, valid):
        """Valid points -> (per-group features [G,C], centers [G,3], mean, scale) in the unit-sphere
        frame the backbone saw (so score can map pixels back to groups in that same frame)."""
        pts = torch.from_numpy(xyz[valid]).float().to(self.dev)
        if len(pts) > self.cfg.n_points:
            pts = pts[torch.randperm(len(pts), device=self.dev)[:self.cfg.n_points]]
        pn, mean, scale = self._norm(pts)
        f, c = Pointmae.pointmae_features(self.net, pn.unsqueeze(0))
        return f[0], c[0], mean, scale

    def fit(self, cat):
        _, train = store.Store.arrays(cat, "train", 0, self.size)
        feats = torch.cat([self._features(a["xyz"], a["valid"])[0] for a in train])   # (sum G, C)
        return Coreset.greedy_coreset(feats, max(1, int(len(feats) * self.cfg.coreset)), 0)

    def _amap(self, bank, xyz, valid):
        f, centers, mean, scale = self._features(xyz, valid)
        gd = Coreset.bank_nn_dist(f, bank)                                             # per-group anomaly (G,)
        pn = (torch.from_numpy(xyz[valid]).float().to(self.dev) - mean) / scale        # all valid pts, same frame
        nearest = PointmaeGroup.square_distance(pn.unsqueeze(0), centers.unsqueeze(0))[0].argmin(1)  # each px->group
        amap = np.zeros(valid.shape, np.float32)
        amap[valid] = gd[nearest].cpu().numpy()
        return amap

    def score(self, bank, cat):
        dft, test = store.Store.arrays(cat, "test", size=self.size)
        amaps = np.stack([self._amap(bank, a["xyz"], a["valid"]) for a in test])
        valids = np.stack([a["valid"].astype(bool) for a in test])
        masks = np.stack([(a["gt"] > 0) for a in test])
        scores = scoring.image_scores(amaps, valids)
        return (amaps, valids, masks, scores,
                dft["label"].to_numpy(), np.array(dft["defect"].to_list()))


SPEC = method_spec("pointmae", PointMAEMethod, PointMAECfg)
