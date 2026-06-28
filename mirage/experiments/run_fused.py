"""Fused PatchCore(rgb) + BTF(FPFH geometry) anomaly maps -> aggregate (M3DM-lite).

Score each test sample with both banks, z-score each map family over the valid test pixels, sum,
then evaluate. The rgb + 3D fusion that closes part of the gap to SOTA.

Run:  python -m mirage.training.run_fused [--cats bagel ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from mirage.data import mvtec, store
from mirage.data.dataset import load_split
from mirage.evaluation import metrics, scoring
from mirage.models.fpfh_bank import FpfhBank
from mirage.models.patchcore import PatchCore


def _arrays(cat, split, label=None):
    df = store.load().filter(pl.col("category") == cat).filter(pl.col("split") == split)
    if label is not None:
        df = df.filter(pl.col("label") == label)
    return df, [store.load_arrays(p) for p in df["path"].to_list()]


def _zscore(maps, valids):
    v = maps[valids]
    return (maps - v.mean()) / (v.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/fused"))
    args = ap.parse_args()

    cats = args.cats or mvtec.categories()
    rows = []
    for c in cats:
        # rgb PatchCore
        train_rgb = load_split(split="train", label=0, cats=[c], channels=["rgb"], device="cuda")
        test_rgb = load_split(split="test", cats=[c], channels=["rgb"], device="cuda")
        pc_maps = PatchCore().fit(train_rgb.x).score_maps(test_rgb.x)
        valids = test_rgb.valid.squeeze(1).cpu().numpy().astype(bool)
        masks = test_rgb.gt.squeeze(1).cpu().numpy().astype(bool)
        labels = test_rgb.df["label"].to_numpy()
        # FPFH geometry
        _, train_arr = _arrays(c, "train", 0)
        bank = FpfhBank().fit([(a["xyz"], a["valid"]) for a in train_arr])
        _, test_arr = _arrays(c, "test")
        fp_maps = np.stack([bank.score_map(a["xyz"], a["valid"]) for a in test_arr])
        # fuse (z-score each family over valid pixels, sum)
        fused = _zscore(pc_maps, valids) + _zscore(fp_maps, valids)
        fused = fused * valids
        scores = scoring.image_scores(fused, valids)
        r = {"category": c, "n": int(len(scores)),
             "img_auroc": metrics.image_auroc(scores, labels),
             "au_pro": metrics.au_pro(fused, masks, valids)}
        rows.append(r)
        print(f"  {c:12s}  img_auroc {r['img_auroc']:.3f}   au_pro {r['au_pro']:.3f}")

    auroc = float(np.nanmean([r["img_auroc"] for r in rows]))
    aupro = float(np.nanmean([r["au_pro"] for r in rows]))
    print(f"  {'MEAN':12s}  img_auroc {auroc:.3f}   au_pro {aupro:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aggregate.json").write_text(
        json.dumps({"method": "fused_rgb_fpfh", "per_category": rows,
                    "mean": {"img_auroc": auroc, "au_pro": aupro}}, indent=2))


if __name__ == "__main__":
    main()
