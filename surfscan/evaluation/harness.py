"""The single per-category eval spine — every method (trained recon, memory bank, handcrafted)
is a (fit_fn, score_fn) pair fed through it.

    fit_fn(cat) -> state
    score_fn(state, cat) -> (amaps, valids, masks, scores, labels, defects)   # all np arrays

`aggregate` is pure (compute per-category + mean + per-defect, no IO — unit-testable).
`run` calls aggregate, then logs everything to MLflow: a fresh method run, or — pass run_id —
back into an existing run (e.g. a training run, so its test metrics live next to its loss curves).
Browse with `mlflow ui`.
"""
from __future__ import annotations

import numpy as np

from surfscan import tracking
from surfscan.data import mvtec
from surfscan.evaluation import diagnostics, metrics

_KEYS = ("amaps", "masks", "valids", "scores", "labels", "defects")


def aggregate(method, fit_fn, score_fn, cats):
    """Pure: run the method over categories, compute the metrics. No MLflow, no side effects."""
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

    return {"method": method, "per_category": rows,
            "mean": {"img_auroc": auroc, "au_pro": aupro}, "per_defect": defect_rows}


def _log(res):
    rows = res["per_category"]
    tracking.metrics({"img_auroc_mean": res["mean"]["img_auroc"], "au_pro_mean": res["mean"]["au_pro"]})
    tracking.per_group("au_pro", {r["category"]: r["au_pro"] for r in rows})
    tracking.per_group("img_auroc", {r["category"]: r["img_auroc"] for r in rows})
    tracking.per_group("au_pro_defect", {d["defect"]: d["au_pro"] for d in res["per_defect"]})
    tracking.artifact_json("aggregate.json", res)


def run(method, fit_fn, score_fn, cats=None, run_id=None, params=None):
    cats = cats or mvtec.categories()
    res = aggregate(method, fit_fn, score_fn, cats)
    if run_id:                                       # log into an existing run (e.g. the train run)
        with tracking.resume(run_id):
            _log(res)
    else:
        with tracking.run("surfscan", method, params=params or {"method": method, "cats": ",".join(cats)}):
            _log(res)
    return res
