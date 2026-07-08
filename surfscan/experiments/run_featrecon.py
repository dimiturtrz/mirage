"""Feature-space reconstruction (RD4AD-lite) over categories -> aggregate. Uses the eval harness.

Per category: extract frozen-backbone features for train-good, train a small AE to reconstruct
them, score the test split by per-location feature-reconstruction error.

Run:  python -m surfscan.experiments.run_featrecon [--cats bagel ...] [--epochs 100]
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import optim

from core.compute import pick_device
from core.data.dataset import load_split
from core.method import Method
from surfscan.evaluation import harness, scoring
from surfscan.models.feat_recon import FeatAE, FeatExtractor


@torch.no_grad()
def _feats(ext, x, batch=32):
    return torch.cat([ext(x[i:i + batch]).float() for i in range(0, len(x), batch)])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    dev = pick_device()
    ext = FeatExtractor(device=dev)

    def fit(c):
        train = load_split(split="train", label=0, cats=[c], channels=["rgb"], device=dev)
        tf = _feats(ext, train.x)
        ae = FeatAE(ch=tf.shape[1]).to(dev)
        opt = optim.AdamW(ae.parameters(), lr=2e-3)
        for _ in range(args.epochs):
            ae.train()
            idx = torch.randperm(len(tf), device=dev)
            for i in range(0, len(tf), 32):
                b = tf[idx[i:i + 32]]
                loss = F.mse_loss(ae(b), b)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return ae

    def score(ae, c):
        test = load_split(split="test", cats=[c], channels=["rgb"], device=dev)
        H, W = test.x.shape[-2:]
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        amaps = _score_maps(ext, ae, test.x, (H, W)) * valids
        return scoring.score_arrays(amaps, test)

    harness.run("feat_recon", Method(fit, score), cats=args.cats)


if __name__ == "__main__":
    main()
