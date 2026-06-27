"""Feature-space reconstruction (RD4AD-lite) over categories -> aggregate.

Per category: extract frozen-backbone features for train-good, train a small AE to reconstruct
them, score the test split by per-location feature-reconstruction error. Tests whether
reconstruction works in feature space (where pixel-space reconstruction failed).

Run:  python -m mirage.training.run_featrecon [--cats bagel ...] [--epochs 100]
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
from mirage.models.feat_recon import FeatAE, FeatExtractor


@torch.no_grad()
def _feats(ext, x, batch=32):
    return torch.cat([ext(x[i:i + batch]).float() for i in range(0, len(x), batch)])


def _score_maps(ext, ae, x, H, W, batch=32):
    out = []
    ae.eval()
    for i in range(0, len(x), batch):
        f = ext(x[i:i + batch]).float()
        with torch.no_grad():
            r = ae(f)
        err = ((f - r) ** 2).sum(1, keepdim=True)                 # n,1,h,w
        m = F.interpolate(err, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        out.append(m.cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("runs/featrecon"))
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    dev = "cuda"
    ext = FeatExtractor(device=dev)

    cats = args.cats or mvtec.categories()
    rows = []
    for c in cats:
        train = load_split(split="train", label=0, cats=[c], channels=["rgb"], device=dev)
        test = load_split(split="test", cats=[c], channels=["rgb"], device=dev)
        tf = _feats(ext, train.x)                                  # N,C,h,w
        ae = FeatAE(ch=tf.shape[1]).to(dev)
        opt = optim.AdamW(ae.parameters(), lr=2e-3)
        for ep in range(args.epochs):
            ae.train()
            idx = torch.randperm(len(tf), device=dev)
            for i in range(0, len(tf), 32):
                b = tf[idx[i:i + 32]]
                loss = F.mse_loss(ae(b), b)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        H, W = test.x.shape[-2:]
        amaps = _score_maps(ext, ae, test.x, H, W)
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        masks = test.gt.squeeze(1).cpu().numpy().astype(bool)
        amaps = amaps * valids
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
        json.dumps({"method": "feat_recon", "per_category": rows,
                    "mean": {"img_auroc": auroc, "au_pro": aupro}}, indent=2))


if __name__ == "__main__":
    main()
