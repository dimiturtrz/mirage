"""FPFH geometry memory bank = BTF (Back to the Feature) — the geometry SOTA-floor + the 3D half.

Per-point FPFH descriptors (Open3D, handcrafted geometric features) from normal samples -> a memory
bank -> test point nearest-neighbor distance = anomaly. The geometry analog of PatchCore: compare
local-surface descriptors to normal. See learning/2026-06-25_fpfh-and-neighborhoods.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from core.compute import Compute


@dataclass(frozen=True)
class FpfhCfg:
    normal_knn: int = 30
    fpfh_knn: int = 100
    clean: bool = False
    graze: float = 0.3


class FpfhBank:
    @staticmethod
    def _normals(pts: Float[np.ndarray, "n 3"], knn: int) -> o3d.geometry.PointCloud:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
        pc.orient_normals_towards_camera_location((0.0, 0.0, 0.0))
        return pc

    @staticmethod
    def fpfh_for_sample(
        xyz: Float[np.ndarray, "h w 3"],
        valid: Bool[np.ndarray, "h w"],
        cfg: FpfhCfg | None = None,
    ) -> tuple[Float[np.ndarray, "n 33"], Int[np.ndarray, "n 2"]]:
        cfg = cfg or FpfhCfg()
        """-> (n,33) FPFH descriptors + (n,2) pixel (row,col) for each valid point.

        clean: drop grazing-angle points (surface nearly perpendicular to the view ray) — that's the
        sparse/noisy periphery that gives garbage normals + FPFH. Recompute normals on the clean set.
        """
        coords = np.argwhere(valid)
        pts = xyz[valid].astype(np.float64)
        pc = FpfhBank._normals(pts, cfg.normal_knn)
        if cfg.clean:
            n = np.asarray(pc.normals)
            view = -pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-9)   # scanner at origin
            keep = np.abs((n * view).sum(1)) > cfg.graze                            # ~1 = facing, ~0 = grazing
            pts, coords = pts[keep], coords[keep]
            pc = FpfhBank._normals(pts, cfg.normal_knn)
        f = o3d.pipelines.registration.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamKNN(cfg.fpfh_knn))
        return np.asarray(f.data).T.astype(np.float32), coords

    @staticmethod
    def _fpfh_batch(
        samples: Any, cfg: FpfhCfg | None = None,
    ) -> list[tuple[Float[np.ndarray, "n 33"], Int[np.ndarray, "n 2"]]]:
        """fpfh_for_sample over samples. Serial by measurement: open3d's normals/FPFH hold the GIL, so a
        ThreadPool gave 1.0x (no speedup, benchmarked on real data); a ProcessPool's Windows-spawn +
        per-sample point-cloud pickle overhead doesn't pay for these small clouds. A real speedup needs a
        vectorized/GPU FPFH, not a pool — out of scope. Kept as a batch helper (fit + score_maps)."""
        return [FpfhBank.fpfh_for_sample(xyz, valid, cfg) for xyz, valid in samples]

    def __init__(self, device: str = "cuda", per_sample: int = 2000, coreset: float = 0.25, seed: int = 0):
        self.device = device
        self.per_sample = per_sample      # cap FPFH per training sample (keeps the bank tractable)
        self.coreset = coreset
        self.seed = seed
        self.bank: Tensor | None = None

    def fit(self, samples: Any) -> FpfhBank:               # samples: list of (xyz, valid)
        rng = np.random.RandomState(self.seed)
        feats = []
        for f, _ in FpfhBank._fpfh_batch(samples):
            if self.per_sample and len(f) > self.per_sample:
                f = f[rng.choice(len(f), self.per_sample, replace=False)]
            feats.append(f)
        feats = np.concatenate(feats)
        if self.coreset < 1.0:
            feats = feats[rng.choice(len(feats), int(len(feats) * self.coreset), replace=False)]
        self.bank = torch.from_numpy(feats).to(self.device)
        return self

    @torch.no_grad()
    def _score_one(
        self, f: Float[np.ndarray, "n 33"], coords: Int[np.ndarray, "n 2"], valid: Bool[np.ndarray, "h w"],
        chunk: int,
    ) -> Float[np.ndarray, "h w"]:
        ft = torch.from_numpy(f).to(self.device)
        d = Compute.batched_forward(
            lambda i0, i1: torch.cdist(ft[i0:i1], self.bank).min(1).values, len(ft), chunk)
        amap = np.zeros(valid.shape, np.float32)
        amap[coords[:, 0], coords[:, 1]] = d.cpu().numpy()
        return amap

    def score_map(
        self, xyz: Float[np.ndarray, "h w 3"], valid: Bool[np.ndarray, "h w"], chunk: int = 4096,
    ) -> Float[np.ndarray, "h w"]:
        f, coords = FpfhBank.fpfh_for_sample(xyz, valid)
        return self._score_one(f, coords, valid, chunk)

    def score_maps(self, samples: Any, chunk: int = 4096) -> Float[np.ndarray, "n h w"]:
        """Batch score -> (N,H,W): FPFH per sample (see _fpfh_batch), then GPU nearest-bank distance.
        Identical to np.stack([score_map(x, v) for x, v in samples]) — a convenience, not a speedup."""
        return np.stack([self._score_one(f, coords, valid, chunk)
                         for (f, coords), (_, valid) in zip(FpfhBank._fpfh_batch(samples), samples, strict=True)])
