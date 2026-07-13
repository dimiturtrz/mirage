"""The single per-category eval spine — every method (trained recon, memory bank, handcrafted)
is a (fit_fn, score_fn) pair fed through it.

    fit_fn(cat) -> state
    score_fn(state, cat) -> ScoreArrays   # named per-sample outputs (see evaluation/method.py)

`aggregate` is pure (compute per-category + mean + per-defect, no IO — unit-testable).
`run` calls aggregate, then logs everything to MLflow: a fresh method run, or — pass run_id —
back into an existing run (e.g. a training run, so its test metrics live next to its loss curves).
Browse with `mlflow ui`.
"""
from __future__ import annotations

import numpy as np

from core.data import mvtec
from core.method import ScoreArrays  # the (fit_fn, score_fn) contract
from core.obs import Obs
from surfscan import tracking
from surfscan.evaluation import diagnostics, metrics

log = Obs.get()


def aggregate(method, fit_fn, score_fn, cats):
    """Pure: run the method over categories, compute the metrics. No MLflow, no side effects."""
    rows = []
    pool: dict[str, list] = {f: [] for f in ScoreArrays._fields}
    for c in cats:
        sa = ScoreArrays(*score_fn(fit_fn(c), c))          # named fields — no fragile tuple order
        rows.append({"category": c, "n": int(len(sa.scores)),
                     "img_auroc": metrics.image_auroc(sa.scores, sa.labels),
                     "au_pro": metrics.au_pro(sa.amaps, sa.masks, sa.valids)})
        log.info(f"  {c:12s}  img_auroc {rows[-1]['img_auroc']:.3f}   au_pro {rows[-1]['au_pro']:.3f}")
        for f in ScoreArrays._fields:
            pool[f].append(getattr(sa, f))

    auroc = float(np.nanmean([r["img_auroc"] for r in rows]))
    aupro = float(np.nanmean([r["au_pro"] for r in rows]))
    log.info(f"  {'MEAN':12s}  img_auroc {auroc:.3f}   au_pro {aupro:.3f}")

    pooled = ScoreArrays(**{f: np.concatenate(pool[f]) for f in ScoreArrays._fields})
    defect_rows = diagnostics.by_defect(pooled)
    log.info("  --- per defect type ---")
    for d in defect_rows:
        log.info(f"  {d['defect']:14s}  img_auroc {d['img_auroc']:.3f}   au_pro {d['au_pro']:.3f}  (n={d['n']})")

    # calibration (ECE) — only meaningful when amaps are probabilities in [0,1] (e.g. the triad's
    # sigmoid outputs); residual/distance methods aren't calibrated, so skip them cleanly.
    amaps_all = pooled.amaps
    calib = None
    if amaps_all.size and np.nanmin(amaps_all) >= 0.0 and np.nanmax(amaps_all) <= 1.0:
        calib = metrics.ece(amaps_all, np.concatenate(pool["masks"]), np.concatenate(pool["valids"]))
        log.info(f"  ECE (calibration under shift) {calib:.4f}")

    return {"method": method, "per_category": rows,
            "mean": {"img_auroc": auroc, "au_pro": aupro}, "ece": calib, "per_defect": defect_rows}


def _log(res):  # pragma: no cover  mlflow metric/artifact logging; aggregate is the pure core
    rows = res["per_category"]
    tracking.metrics({"img_auroc_mean": res["mean"]["img_auroc"], "au_pro_mean": res["mean"]["au_pro"]})
    if res.get("ece") is not None:
        tracking.metrics({"ece": res["ece"]})
    tracking.per_group("au_pro", {r["category"]: r["au_pro"] for r in rows})
    tracking.per_group("img_auroc", {r["category"]: r["img_auroc"] for r in rows})
    tracking.per_group("au_pro_defect", {d["defect"]: d["au_pro"] for d in res["per_defect"]})
    tracking.artifact_json("aggregate.json", res)


def run(method, m, cats=None, run_id=None, params=None):  # pragma: no cover  mlflow wrapper; aggregate is pure
    cats = cats or mvtec.Mvtec.categories()
    res = aggregate(method, m.fit, m.score, cats)
    if run_id:                                       # log into an existing run (e.g. the train run)
        with tracking.resume(run_id):
            _log(res)
    else:
        with tracking.run("surfscan", method, params=params or {"method": method, "cats": ",".join(cats)}):
            _log(res)
    return res
