"""Shared per-category eval harness — runs a method over categories, logs to MLflow.

A method is a (fit_fn, score_fn) pair:
    fit_fn(cat) -> state
    score_fn(state, cat) -> (amaps, valids, masks, scores, labels, defects)   # all np arrays
The harness computes image-AUROC + AU-PRO per category + mean, pools across categories for the
per-defect-type diagnostics, prints, and logs everything to one MLflow run (params + metrics +
per-category + per-defect + the aggregate artifact). Browse with `mlflow ui`.
"""
from __future__ import annotations

import numpy as np

from surfscan import tracking
from surfscan.data import mvtec
from surfscan.evaluation import diagnostics, metrics

_KEYS = ("amaps", "masks", "valids", "scores", "labels", "defects")


def run(method, fit_fn, score_fn, cats=None):
    cats = cats or mvtec.categories()
    with tracking.run("surfscan", method, params={"method": method, "cats": ",".join(cats)}):
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
        tracking.metrics({"img_auroc_mean": auroc, "au_pro_mean": aupro})
        tracking.per_group("au_pro", {r["category"]: r["au_pro"] for r in rows})
        tracking.per_group("img_auroc", {r["category"]: r["img_auroc"] for r in rows})
        tracking.per_group("au_pro_defect", {d["defect"]: d["au_pro"] for d in defect_rows})
        tracking.artifact_json("aggregate.json", res)
        return res
