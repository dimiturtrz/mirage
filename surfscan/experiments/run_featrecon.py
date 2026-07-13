"""Feature-space reconstruction (RD4AD-lite) over categories -> aggregate. Uses the eval harness.

Per category: extract frozen-backbone features for train-good, train a small AE to reconstruct
them, score the test split by per-location feature-reconstruction error. The AE training is the shared
`Trainer`; the method's step_fn is one MSE reconstruction step.

Run:  python -m surfscan.run featrecon [--cats bagel ...] [--epochs 100]
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import optim

from core.data.dataset import GpuSplit
from surfscan.evaluation import scoring
from surfscan.method_cli import MethodCli
from surfscan.models.feat_recon import FeatAE, FeatExtractor
from surfscan.training.trainer import Trainer


@dataclass(frozen=True)
class FeatReconCfg:
    epochs: int = 100


class FeatReconMethod:
    run_name = "feat_recon"

    def __init__(self, cfg: FeatReconCfg, dev):
        self.cfg = cfg
        self.dev = dev
        self.ext = FeatExtractor(device=dev)

    @staticmethod
    @torch.no_grad()
    def _feats(ext, x, batch=32):
        return torch.cat([ext(x[i:i + batch]).float() for i in range(0, len(x), batch)])

    @staticmethod
    def _score_maps(ext, ae, x, hw, batch=32):
        H, W = hw
        out = []
        ae.eval()
        for i in range(0, len(x), batch):
            with torch.no_grad():
                f = ext(x[i:i + batch]).float()
                err = ((f - ae(f)) ** 2).sum(1, keepdim=True)
            m = F.interpolate(err, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            out.append(m.cpu())
        return torch.cat(out).numpy()

    def fit(self, cat):
        train = GpuSplit.load_split(split="train", label=0, cats=[cat], channels=["rgb"], device=self.dev)
        tf = FeatReconMethod._feats(self.ext, train.x)
        ae = FeatAE(ch=tf.shape[1]).to(self.dev)
        opt = optim.AdamW(ae.parameters(), lr=2e-3)

        def step(idx):
            b = tf[idx]
            return F.mse_loss(ae(b), b)

        return Trainer(ae, opt, self.dev, batch=32).fit(len(tf), self.cfg.epochs, step)

    def score(self, ae, cat):
        test = GpuSplit.load_split(split="test", cats=[cat], channels=["rgb"], device=self.dev)
        H, W = test.x.shape[-2:]
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        amaps = FeatReconMethod._score_maps(self.ext, ae, test.x, (H, W)) * valids
        return scoring.Scoring.score_arrays(amaps, test)


SPEC = MethodCli.method_spec("featrecon", FeatReconMethod, FeatReconCfg)
