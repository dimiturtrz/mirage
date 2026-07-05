"""DRAEM over categories -> aggregate. Synthesis-recon, Stage-0, self-contained. Uses the harness.

Per category: train recon+disc on good samples with synthetic proxy anomalies, score the test
split by the discriminator output. Default channel = xyz (the geometry signal).

Run:  python -m surfscan.experiments.run_draem [--cats bagel ...] [--epochs 150] [--channels xyz]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from core.data.dataset import load_split
from core.data.defects import synthesize   # realistic coherent defects (was draem's crude Perlin, 0.48)
from surfscan.evaluation import harness, scoring
from surfscan.models.draem import Draem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--channels", nargs="*", default=["xyz"])
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    dev = "cuda"
    rng = np.random.RandomState(0)

    def fit(c):
        train = load_split(split="train", label=0, cats=[c], channels=args.channels, device=dev)
        model = Draem(ch=train.in_ch).to(dev, memory_format=torch.channels_last)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        n = len(train)
        for _ in range(args.epochs):
            model.train()
            idx = torch.randperm(n, device=dev)
            for i in range(0, n, 16):
                b = idx[i:i + 16]
                x, v = train.x[b], train.valid[b]
                aug, mask = synthesize(x, v, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rec, logits = model(aug.to(memory_format=torch.channels_last))
                    rl = (((rec - x) ** 2) * v).sum() / (v.sum() * x.shape[1] + 1e-8)
                    dl = F.binary_cross_entropy_with_logits(logits.float(), mask, weight=v)
                    loss = rl + dl
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return model

    def score(model, c):
        test = load_split(split="test", cats=[c], channels=args.channels, device=dev)
        model.eval()
        amaps = []
        with torch.no_grad():
            for i in range(0, len(test), 16):
                x, v = test.x[i:i + 16], test.valid[i:i + 16]
                _, logits = model(x.to(memory_format=torch.channels_last))
                amaps.append((torch.sigmoid(logits.float()) * v).squeeze(1).cpu())
        amaps = torch.cat(amaps).numpy()
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        masks = test.gt.squeeze(1).cpu().numpy().astype(bool)
        scores = scoring.image_scores(amaps, valids)
        return (amaps, valids, masks, scores,
                test.df["label"].to_numpy(), np.array(test.df["defect"].to_list()))

    harness.run(f"draem_realistic_{'_'.join(args.channels)}", fit, score, cats=args.cats)


if __name__ == "__main__":
    main()
