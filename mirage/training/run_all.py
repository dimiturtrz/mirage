"""Per-category VAE train+eval over all categories -> aggregate mean.

One model per object is the MVTec convention (BTF/M3DM do this); a single model across
bagel+tire+rope would be the harder multi-category setting. Each run is fast (GPU-resident),
so all 10 take a couple of minutes.

Run:  python -m mirage.training.run_all [--out-root runs/vae_all] [--cats ...] [--epochs 100]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mirage.data import mvtec
from mirage.evaluation.evaluate import evaluate
from mirage.training.hparams import HParams
from mirage.training.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("runs/vae_all"))
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    cats = args.cats or mvtec.categories()
    rows = []
    for c in cats:
        print(f"\n===== {c} =====")
        hp = HParams(cats=[c], epochs=args.epochs, compile=False)   # compile off: recompiling per category isn't worth it
        out = args.out_root / c
        train(hp, out)
        res = evaluate(out, cats=[c])
        rows.append(res["per_category"][0])

    auroc = float(np.nanmean([r["img_auroc"] for r in rows]))
    aupro = float(np.nanmean([r["au_pro"] for r in rows]))
    print("\n===== AGGREGATE (per-category models) =====")
    for r in rows:
        print(f"  {r['category']:12s}  img_auroc {r['img_auroc']:.3f}   au_pro {r['au_pro']:.3f}")
    print(f"  {'MEAN':12s}  img_auroc {auroc:.3f}   au_pro {aupro:.3f}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "aggregate.json").write_text(
        json.dumps({"per_category": rows, "mean": {"img_auroc": auroc, "au_pro": aupro}}, indent=2))


if __name__ == "__main__":
    main()
