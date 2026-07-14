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
from core.obs import Obs, Progress
from surfscan import tracking
from surfscan.evaluation import diagnostics, invariants, metrics, predictions

log = Obs.get()


class Harness:
    """The single per-category eval spine — every method is a (fit_fn, score_fn) pair fed through it."""

    @staticmethod
    def aggregate(method, fit_fn, score_fn, cats):
        """Run the method over categories (per-category fit+score, ETA-logged), then compute the metrics."""
        prog = Progress(len(cats), tag=method)
        by_cat = []
        for c in cats:
            by_cat.append(ScoreArrays(*score_fn(fit_fn(c), c)))        # named fields — no fragile tuple order
            prog.tick(c)                                               # done/total + ETA (spot a slow run live)
        return Harness.compute(method, by_cat, cats)

    @staticmethod
    def from_artifact(npz):
        """Recompute the full result OFFLINE from persisted per-image predictions (Predictions.pack npz) —
        no fit/score, no GPU. A new bootstrap / macro fix / ECE variant becomes a re-`compute`, not a re-run."""
        cats, by_cat = predictions.Predictions.unpack(npz)
        return Harness.compute(f"recompute[{len(cats)}cat]", by_cat, cats)

    @staticmethod
    def compute(method, by_cat, cats):
        """The pure metric/CI core over per-category ScoreArrays — shared by live eval (`aggregate`) and
        offline recompute (`from_artifact`). Every headline number is a function of these arrays alone."""
        rows = []
        pool: dict[str, list] = {f: [] for f in ScoreArrays._fields}
        for c, sa in zip(cats, by_cat, strict=True):
            rows.append({"category": c, "n": len(sa.scores),
                         "img_auroc": metrics.Metrics.image_auroc(sa.scores, sa.labels),
                         "au_pro": metrics.Metrics.au_pro(sa.amaps, sa.masks, sa.valids)})
            log.info(f"  {c:12s}  img_auroc {rows[-1]['img_auroc']:.3f}   au_pro {rows[-1]['au_pro']:.3f}")
            for f in ScoreArrays._fields:
                pool[f].append(getattr(sa, f))

        auroc = float(np.nanmean([r["img_auroc"] for r in rows]))
        aupro = float(np.nanmean([r["au_pro"] for r in rows]))
        log.info(f"  {'MEAN':12s}  img_auroc {auroc:.3f}   au_pro {aupro:.3f}")

        pooled = ScoreArrays(**{f: np.concatenate(pool[f]) for f in ScoreArrays._fields})
        defect_rows = diagnostics.Diagnostics.by_defect(pooled)
        log.info("  --- per defect type ---")
        for d in defect_rows:
            log.info(f"  {d['defect']:14s}  img_auroc {d['img_auroc']:.3f}   au_pro {d['au_pro']:.3f}  (n={d['n']})")

        # uncertainty from THIS run's test set — a stratified bootstrap over the images (not a retrain).
        # MACRO (mean-of-categories) to match the reported point: resample within each category, average.
        auroc_cat = [(sa.scores, sa.labels) for sa in by_cat]
        pro_cat = [(sa.amaps, sa.masks, sa.valids) for sa in by_cat]
        with Progress.stage("bootstrap CI"):                   # the au_pro bootstrap is the usual slow stage
            brackets = {"img_auroc": metrics.Metrics.boot_macro_ci(metrics.Metrics.image_auroc, auroc_cat),
                        "au_pro": metrics.Metrics.boot_macro_ci(metrics.Metrics.au_pro, pro_cat)}
        log.info(f"  img_auroc {brackets['img_auroc'][0]:.3f} "
                 f"[{brackets['img_auroc'][1]:.3f}, {brackets['img_auroc'][2]:.3f}]   "
                 f"au_pro {brackets['au_pro'][0]:.3f} "
                 f"[{brackets['au_pro'][1]:.3f}, {brackets['au_pro'][2]:.3f}]  (95% boot CI)")

        # calibration (ECE) — only meaningful when amaps are probabilities in [0,1] (e.g. the triad's
        # sigmoid outputs); residual/distance methods aren't calibrated, so skip them cleanly.
        amaps_all = pooled.amaps
        calib = None
        if amaps_all.size and np.nanmin(amaps_all) >= 0.0 and np.nanmax(amaps_all) <= 1.0:
            calib = metrics.Metrics.ece(amaps_all, np.concatenate(pool["masks"]), np.concatenate(pool["valids"]))
            log.info(f"  ECE (calibration under shift) {calib:.4f}")

        result = {"method": method, "per_category": rows, "mean": {"img_auroc": auroc, "au_pro": aupro},
               "ci": brackets, "ece": calib, "per_defect": defect_rows, "by_cat": by_cat}
        invariants.Invariants.check(result)                       # loud at run end if a number is inconsistent
        return result

    @staticmethod
    def log(result):  # pragma: no cover  mlflow metric/artifact logging; aggregate is the pure core
        rows = result["per_category"]
        mean = result["mean"]
        tracking.Tracker.metrics({"img_auroc_mean": mean["img_auroc"], "au_pro_mean": mean["au_pro"]})
        brackets = result.get("ci", {})
        for metric, (_point, low, high) in brackets.items():   # bootstrap brackets next to the point metric
            if not np.isnan(low):                               # skip NaN (metric undefined on the run)
                tracking.Tracker.metrics({f"{metric}_lo": low, f"{metric}_hi": high})
        if result.get("ece") is not None:
            tracking.Tracker.metrics({"ece": result["ece"]})
        tracking.Tracker.per_group("au_pro", {row["category"]: row["au_pro"] for row in rows})
        tracking.Tracker.per_group("img_auroc", {row["category"]: row["img_auroc"] for row in rows})
        tracking.Tracker.per_group("au_pro_defect",
                                   {row["defect"]: row["au_pro"] for row in result["per_defect"]})
        tracking.Tracker.artifact_json("aggregate.json",
                                       {key: value for key, value in result.items() if key != "by_cat"})
        cats = [row["category"] for row in rows]                # persist per-image predictions -> offline recompute
        tracking.Tracker.artifact_npz("predictions.npz", predictions.Predictions.pack(result["by_cat"], cats))

    @staticmethod
    def run(method, method_obj, cats=None, run_id=None, params=None):  # pragma: no cover  mlflow wrapper
        cats = cats or mvtec.Mvtec().categories()
        result = Harness.aggregate(method, method_obj.fit, method_obj.score, cats)
        if run_id:                                       # log into an existing run (e.g. the train run)
            with tracking.Tracker.resume(run_id):
                Harness.log(result)
        else:
            with tracking.Tracker.run("surfscan", method, params=params or {"method": method, "cats": ",".join(cats)}):
                Harness.log(result)
        return result
