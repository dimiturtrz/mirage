"""BTF (FPFH geometry memory bank) over categories -> aggregate. The geometry SOTA-floor.

Fit a normal-FPFH bank per category, score the test split through OUR eval harness. No training.

Run:  python -m mirage.training.run_btf [--cats bagel ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from mirage.data import mvtec, store
from mirage.evaluation import metrics, scoring
from mirage.models.fpfh_bank import FpfhBank


def _load(cat, split, label=None):
    df = store.load().filter(pl.col("category") == cat).filter(pl.col("split") == split)
    if label is not None:
        df = df.filter(pl.col("label") == label)
    return df, [store.load_arrays(p) for p in df["path"].to_list()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/btf"))
    args = ap.parse_args()

    cats = args.cats or mvtec.categories()
    rows = []
    for c in cats:
        _, train = _load(c, "train", 0)
        bank = FpfhBank().fit([(a["xyz"], a["valid"]) for a in train])
        dft, test = _load(c, "test")
        amaps = np.stack([bank.score_map(a["xyz"], a["valid"]) for a in test])
        valids = np.stack([a["valid"].astype(bool) for a in test])
        masks = np.stack([(a["gt"] > 0) for a in test])
        scores = scoring.image_scores(amaps, valids)
        labels = dft["label"].to_numpy()
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
        json.dumps({"method": "btf_fpfh", "per_category": rows,
                    "mean": {"img_auroc": auroc, "au_pro": aupro}}, indent=2))


if __name__ == "__main__":
    main()
