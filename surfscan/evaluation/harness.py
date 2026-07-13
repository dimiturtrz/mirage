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


class Harness:
    """The single per-category eval spine — every method is a (fit_fn, score_fn) pair fed through it."""

    @staticmethod
    def aggregate(method, fit_fn, score_fn, cats):
        """Pure: run the method over categories, compute the metrics. No MLflow, no side effects."""
        rows = []
        pool: dict[str, list] = {f: [] for f in ScoreArrays._fields}
        for c in cats:
            sa = ScoreArrays(*score_fn(fit_fn(c), c))          # named fields — no fragile tuple order
            rows.append({"category": c, "n": int(len(sa.scores)),
                         "img_auroc": metrics.Metrics.image_auroc(sa.scores, sa.labels),
                         "au_pro": metrics.Metrics.au_pro(sa.amaps, sa.masks, sa.valids)})
            log.info(f"  {c:12s}  img_auroc {rows[-1]['img_auroc']:.3f}   au_pro {rows[-1]['au_pro']:.3f}")
            for f in ScoreArrays._fields:
                pool[f].append(getattr(sa, f))

        auroc = float(np.nanmean([r["img_auroc"] for r in rows]))
        aupro = float(np.nanmean([r["au_pro"] for r in rows]))
        log.info(f"  {'MEAN':12s}  img_auroc {auroc:.3f}   au_pro {aupro:.3f}")

        by_cat = [ScoreArrays(**{f: pool[f][i] for f in ScoreArrays._fields}) for i in range(len(rows))]
        pooled = ScoreArrays(**{f: np.concatenate(pool[f]) for f in ScoreArrays._fields})
        defect_rows = diagnostics.Diagnostics.by_defect(pooled)
        log.info("  --- per defect type ---")
        for d in defect_rows:
            log.info(f"  {d['defect']:14s}  img_auroc {d['img_auroc']:.3f}   au_pro {d['au_pro']:.3f}  (n={d['n']})")

        # uncertainty from THIS run's test set — a stratified bootstrap over the images (not a retrain).
        # MACRO (mean-of-categories) to match the reported point: resample within each category, average.
        auroc_cat = [(sa.scores, sa.labels) for sa in by_cat]
        pro_cat = [(sa.amaps, sa.masks, sa.valids) for sa in by_cat]
        ci = {"img_auroc": metrics.Metrics.boot_macro_ci(metrics.Metrics.image_auroc, auroc_cat),
              "au_pro": metrics.Metrics.boot_macro_ci(metrics.Metrics.au_pro, pro_cat)}
        log.info(f"  img_auroc {ci['img_auroc'][0]:.3f} [{ci['img_auroc'][1]:.3f}, {ci['img_auroc'][2]:.3f}]   "
                 f"au_pro {ci['au_pro'][0]:.3f} [{ci['au_pro'][1]:.3f}, {ci['au_pro'][2]:.3f}]  (95% boot CI)")

        # calibration (ECE) — only meaningful when amaps are probabilities in [0,1] (e.g. the triad's
        # sigmoid outputs); residual/distance methods aren't calibrated, so skip them cleanly.
        amaps_all = pooled.amaps
        calib = None
        if amaps_all.size and np.nanmin(amaps_all) >= 0.0 and np.nanmax(amaps_all) <= 1.0:
            calib = metrics.Metrics.ece(amaps_all, np.concatenate(pool["masks"]), np.concatenate(pool["valids"]))
            log.info(f"  ECE (calibration under shift) {calib:.4f}")

        return {"method": method, "per_category": rows, "mean": {"img_auroc": auroc, "au_pro": aupro},
                "ci": ci, "ece": calib, "per_defect": defect_rows, "by_cat": by_cat}

    @staticmethod
    def _log(res):  # pragma: no cover  mlflow metric/artifact logging; aggregate is the pure core
        rows = res["per_category"]
        tracking.Tracker.metrics({"img_auroc_mean": res["mean"]["img_auroc"], "au_pro_mean": res["mean"]["au_pro"]})
        ci = res.get("ci", {})
        for metric, (_p, lo, hi) in ci.items():                # bootstrap brackets next to the point metric
            if lo == lo:                                        # skip NaN (metric undefined on the run)
                tracking.Tracker.metrics({f"{metric}_lo": lo, f"{metric}_hi": hi})
        if res.get("ece") is not None:
            tracking.Tracker.metrics({"ece": res["ece"]})
        tracking.Tracker.per_group("au_pro", {r["category"]: r["au_pro"] for r in rows})
        tracking.Tracker.per_group("img_auroc", {r["category"]: r["img_auroc"] for r in rows})
        tracking.Tracker.per_group("au_pro_defect", {d["defect"]: d["au_pro"] for d in res["per_defect"]})
        tracking.Tracker.artifact_json("aggregate.json", {k: v for k, v in res.items() if k != "by_cat"})

    @staticmethod
    def run(method, m, cats=None, run_id=None, params=None):  # pragma: no cover  mlflow wrapper; aggregate is pure
        cats = cats or mvtec.Mvtec().categories()
        res = Harness.aggregate(method, m.fit, m.score, cats)
        if run_id:                                       # log into an existing run (e.g. the train run)
            with tracking.Tracker.resume(run_id):
                Harness._log(res)
        else:
            with tracking.Tracker.run("surfscan", method, params=params or {"method": method, "cats": ",".join(cats)}):
                Harness._log(res)
        return res
