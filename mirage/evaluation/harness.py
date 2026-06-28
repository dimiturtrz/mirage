"""Shared per-category eval harness — removes the duplicated loop from the run_* scripts.

A method is a (fit_fn, score_fn) pair:
    fit_fn(cat) -> state                                  # fit/train on that category's good split
    score_fn(state, cat) -> (amaps, valids, masks, scores, labels, defects)  # all np arrays
The harness loops categories, computes image-AUROC + AU-PRO per category + mean, pools across
categories for the per-defect-type diagnostics, prints, and saves aggregate.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mirage.data import mvtec
from mirage.evaluation import diagnostics, metrics

_KEYS = ("amaps", "masks", "valids", "scores", "labels", "defects")


def run(method, fit_fn, score_fn, cats=None, out=None):
    cats = cats or mvtec.categories()
    rows = []
    pool = {k: [] for k in _KEYS}
    for c in cats:
        amaps, valids, masks, scores, labels, defects = score_fn(fit_fn(c), c)
        rows.append({"category": c, "n": int(len(scores)),
                     "img_auroc": metrics.image_auroc(scores, labels),
                     "au_pro": metrics.au_pro(amaps, masks, valids)})
        print(f"  {c:12s}  img_auroc {rows[-1]['img_auroc']:.3f}   au_pro {rows[-1]['au_pro']:.3f}")
        for k, v in zip(_KEYS, (amaps, masks, valids, scores, labels, defects)):
            pool[k].append(v)

    auroc = float(np.nanmean([r["img_auroc"] for r in rows]))
    aupro = float(np.nanmean([r["au_pro"] for r in rows]))
    print(f"  {'MEAN':12s}  img_auroc {auroc:.3f}   au_pro {aupro:.3f}")

    defect_rows = diagnostics.by_defect(
        np.concatenate(pool["amaps"]), np.concatenate(pool["scores"]),
        np.concatenate(pool["masks"]), np.concatenate(pool["valids"]),
        np.concatenate(pool["labels"]), np.concatenate(pool["defects"]))
    print("  --- per defect type ---")
    for d in defect_rows:
        print(f"  {d['defect']:14s}  img_auroc {d['img_auroc']:.3f}   au_pro {d['au_pro']:.3f}  (n={d['n']})")

    res = {"method": method, "per_category": rows,
           "mean": {"img_auroc": auroc, "au_pro": aupro}, "per_defect": defect_rows}
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "aggregate.json").write_text(json.dumps(res, indent=2))
    return res
