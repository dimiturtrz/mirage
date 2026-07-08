"""FPFH geometry memory bank = BTF (Back to the Feature) — the geometry SOTA-floor + the 3D half.

Per-point FPFH descriptors (Open3D, handcrafted geometric features) from normal samples -> a memory
bank -> test point nearest-neighbor distance = anomaly. The geometry analog of PatchCore: compare
local-surface descriptors to normal. See learning/2026-06-25_fpfh-and-neighborhoods.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch


def _normals(pts, knn):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
    pc.orient_normals_towards_camera_location((0.0, 0.0, 0.0))
    return pc


@dataclass(frozen=True)
class FpfhCfg:
    normal_knn: int = 30
    fpfh_knn: int = 100
    clean: bool = False
    graze: float = 0.3


def fpfh_for_sample(xyz, valid, cfg=None):
    cfg = cfg or FpfhCfg()
    """-> (n,33) FPFH descriptors + (n,2) pixel (row,col) for each valid point.

    clean: drop grazing-angle points (surface nearly perpendicular to the view ray) — that's the
    sparse/noisy periphery that gives garbage normals + FPFH. Recompute normals on the clean set.
    """
    coords = np.argwhere(valid)
    pts = xyz[valid].astype(np.float64)
    pc = _normals(pts, cfg.normal_knn)
    if cfg.clean:
        n = np.asarray(pc.normals)
        view = -pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)   # scanner at origin
        keep = np.abs((n * view).sum(1)) > cfg.graze                            # ~1 = facing, ~0 = grazing
        pts, coords = pts[keep], coords[keep]
        pc = _normals(pts, cfg.normal_knn)
    f = o3d.pipelines.registration.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamKNN(cfg.fpfh_knn))
    return np.asarray(f.data).T.astype(np.float32), coords


class FpfhBank:
    def __init__(self, device="cuda", per_sample=2000, coreset=0.25, seed=0):
        self.device = device
        self.per_sample = per_sample      # cap FPFH per training sample (keeps the bank tractable)
        self.coreset = coreset
        self.seed = seed
        self.bank = None

    def fit(self, samples):               # samples: list of (xyz, valid)
        rng = np.random.RandomState(self.seed)
        feats = []
        for xyz, valid in samples:
            f, _ = fpfh_for_sample(xyz, valid)
            if self.per_sample and len(f) > self.per_sample:
                f = f[rng.choice(len(f), self.per_sample, replace=False)]
            feats.append(f)
        feats = np.concatenate(feats)
        if self.coreset < 1.0:
            feats = feats[rng.choice(len(feats), int(len(feats) * self.coreset), replace=False)]
        self.bank = torch.from_numpy(feats).to(self.device)
        return self

    @torch.no_grad()
    def score_map(self, xyz, valid, chunk=4096):
        f, coords = fpfh_for_sample(xyz, valid)
        ft = torch.from_numpy(f).to(self.device)
        d = torch.empty(len(ft), device=self.device)
        for i in range(0, len(ft), chunk):
            d[i:i + chunk] = torch.cdist(ft[i:i + chunk], self.bank).min(1).values
        amap = np.zeros(valid.shape, np.float32)
        amap[coords[:, 0], coords[:, 1]] = d.cpu().numpy()
        return amap
