"""Feature-space reconstruction (RD4AD-lite) over categories -> aggregate. Uses the eval harness.

Per category: extract frozen-backbone features for train-good, train a small AE to reconstruct
them, score the test split by per-location feature-reconstruction error. The AE training is the shared
`Trainer`; the method's step_fn is one MSE reconstruction step.

Run:  python -m surfscan.run featrecon [--cats bagel ...] [--epochs 100]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, optim

from core.compute import Compute
from core.data.static.dataset import GpuSplit
from core.data.static.mvtec import Split
from core.method import ScoreArrays
from surfscan.evaluation import scoring
from surfscan.method_cli import MethodCli
from surfscan.models.feat_recon import FeatAE, FeatExtractor
from surfscan.training.trainer import Trainer


@dataclass(frozen=True)
class FeatReconCfg:
    epochs: int = 100


class FeatReconMethod:
    run_name = "feat_recon"

    def __init__(self, cfg: FeatReconCfg, dev: str | torch.device) -> None:
        self.cfg = cfg
        self.dev = dev
        self.ext = FeatExtractor(device=dev)

    @staticmethod
    @torch.no_grad()
    def _feats(
        ext: FeatExtractor, x: Float[Tensor, "n c h w"], batch: int = 32
    ) -> Float[Tensor, "n d"]:
        return Compute.batched_forward(lambda i0, i1: ext(x[i0:i1]).float(), len(x), batch)

    @staticmethod
    def _score_maps(
        ext: FeatExtractor, ae: nn.Module, x: Float[Tensor, "n c h w"], hw: tuple[int, int],
        batch: int = 32
    ) -> Float[np.ndarray, "n h w"]:
        H, W = hw
        ae.eval()

        def span(i0: int, i1: int) -> Float[Tensor, "b h w"]:
            with torch.no_grad():
                f = ext(x[i0:i1]).float()
                err = ((f - ae(f)) ** 2).sum(1, keepdim=True)
            m = F.interpolate(err, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            return m.cpu()
        return Compute.batched_forward(span, len(x), batch).numpy()

    def fit(self, cat: str) -> nn.Module:
        train = GpuSplit.load_split(split=Split.TRAIN, label=0, cats=[cat], channels=["rgb"], device=self.dev)
        tf = FeatReconMethod._feats(self.ext, train.x)
        ae = FeatAE(ch=tf.shape[1]).to(self.dev)
        opt = optim.AdamW(ae.parameters(), lr=2e-3)

        def step(idx: Tensor) -> Float[Tensor, ""]:
            b = tf[idx]
            return F.mse_loss(ae(b), b)

        return Trainer(ae, opt, self.dev, batch=32).fit(len(tf), self.cfg.epochs, step)

    def score(self, ae: nn.Module, cat: str) -> ScoreArrays:
        test = GpuSplit.load_split(split=Split.TEST, cats=[cat], channels=["rgb"], device=self.dev)
        H, W = test.x.shape[-2:]
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        amaps = FeatReconMethod._score_maps(self.ext, ae, test.x, (H, W)) * valids
        return scoring.Scoring.score_arrays(amaps, test)


SPEC = MethodCli.method_spec("featrecon", FeatReconMethod, FeatReconCfg)
