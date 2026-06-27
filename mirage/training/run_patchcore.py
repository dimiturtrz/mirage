"""PatchCore feature-memory-bank over all categories -> aggregate (no training).

Fit a normal-feature bank per category on rgb, score the test split through OUR eval harness
(image AUROC + pixel AU-PRO). This is the "compare to normal" paradigm vs the reconstruction floor.

Run:  python -m mirage.training.run_patchcore [--cats bagel ...] [--coreset 0.1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mirage.data import mvtec
from mirage.data.dataset import load_split
from mirage.evaluation import metrics, scoring
from mirage.models.patchcore import PatchCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("runs/patchcore"))
    args = ap.parse_args()

    cats = args.cats or mvtec.categories()
    rows = []
    for c in cats:
        train = load_split(split="train", label=0, cats=[c], channels=["rgb"], device="cuda")
        test = load_split(split="test", cats=[c], channels=["rgb"], device="cuda")
        pc = PatchCore().fit(train.x, coreset=args.coreset)
        amaps = pc.score_maps(test.x)
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
        json.dumps({"method": "patchcore_rgb", "per_category": rows,
                    "mean": {"img_auroc": auroc, "au_pro": aupro}}, indent=2))


if __name__ == "__main__":
    main()
