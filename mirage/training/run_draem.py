"""DRAEM over categories -> aggregate. Synthesis-recon, Stage-0, self-contained.

Per category: train recon+disc on good samples with synthetic proxy anomalies, score the test
split by the discriminator output. Default channel = xyz (the geometry signal).

Run:  python -m mirage.training.run_draem [--cats bagel ...] [--epochs 150] [--channels xyz]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from mirage.data import mvtec
from mirage.data.dataset import load_split
from mirage.evaluation import metrics, scoring
from mirage.models.draem import Draem, synthesize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--channels", nargs="*", default=["xyz"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("runs/draem"))
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    dev = "cuda"
    rng = np.random.RandomState(0)

    cats = args.cats or mvtec.categories()
    rows = []
    for c in cats:
        train = load_split(split="train", label=0, cats=[c], channels=args.channels, device=dev)
        test = load_split(split="test", cats=[c], channels=args.channels, device=dev)
        model = Draem(ch=train.in_ch).to(dev, memory_format=torch.channels_last)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        n = len(train)
        for ep in range(args.epochs):
            model.train()
            idx = torch.randperm(n, device=dev)
            for i in range(0, n, 16):
                b = idx[i:i + 16]
                x = train.x[b]; v = train.valid[b]
                aug, mask = synthesize(x, v, rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rec, logits = model(aug.to(memory_format=torch.channels_last))
                    rl = (((rec - x) ** 2) * v).sum() / (v.sum() * x.shape[1] + 1e-8)
                    dl = F.binary_cross_entropy_with_logits(logits.float(), mask, weight=v)
                    loss = rl + dl
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        # score
        model.eval()
        amaps = []
        with torch.no_grad():
            for i in range(0, len(test), 16):
                x = test.x[i:i + 16]; v = test.valid[i:i + 16]
                _, logits = model(x.to(memory_format=torch.channels_last))
                amaps.append((torch.sigmoid(logits.float()) * v).squeeze(1).cpu())
        amaps = torch.cat(amaps).numpy()
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        masks = test.gt.squeeze(1).cpu().numpy().astype(bool)
        scores = scoring.image_scores(amaps, valids)
        labels = test.df["label"].to_numpy()
        r = {"category": c, "n": int(len(scores)),
             "img_auroc": metrics.image_auroc(scores, labels),
             "au_pro": metrics.au_pro(amaps, masks, valids)}
        rows.append(r)
        print(f"  {c:12s}  img_auroc {r['img_auroc']:.3f}   au_pro {r['au_pro']:.3f}")

    auroc = float(np.nanmean([r["img_auroc"] for r in rows]))
    aupro = float(np.nanmean([r["au_pro"] for r in rows]))
    print(f"  {'MEAN':12s}  img_auroc {auroc:.3f}   au_pro {aupro:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aggregate.json").write_text(
        json.dumps({"method": f"draem_{'_'.join(args.channels)}", "per_category": rows,
                    "mean": {"img_auroc": auroc, "au_pro": aupro}}, indent=2))


if __name__ == "__main__":
    main()
