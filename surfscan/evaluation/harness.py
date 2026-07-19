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

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TypeAlias, TypedDict, cast

import numpy as np

from core import invariants, metrics
from core.data.static import mvtec
from core.method import AnomalyMethod, FitFn, Method, ScoreArrays, ScoreFn, StateT  # the (fit_fn, score_fn) contract
from core.obs import Obs, Progress
from surfscan import tracking
from surfscan.evaluation import diagnostics, predictions
from surfscan.tracking import RunParams

log = Obs.get()

Bracket: TypeAlias = tuple[float, float, float]   # (point, lo, hi) — one bootstrap interval


class CategoryRow(TypedDict):
    """One per-category row of the eval table (`Metrics.image_auroc` / `Metrics.au_pro` over that split)."""

    category: str
    n: int
    img_auroc: float
    au_pro: float


class DefectRow(TypedDict):
    """One per-defect-type diagnostic row (that type's anomalies scored against ALL normals)."""

    defect: str
    n: int
    img_auroc: float
    au_pro: float


@dataclass(frozen=True)
class Scores:
    """The two headline metrics that always travel together (mean over categories / defects)."""

    img_auroc: float
    au_pro: float

    @classmethod
    def macro(cls, rows: list[CategoryRow]) -> Scores:
        """Macro headline (mean over categories) from per-category rows carrying `img_auroc` / `au_pro`
        — the nanmean reduction shared by the live harness and the per-category run aggregator."""
        return cls(float(np.nanmean([r["img_auroc"] for r in rows])),
                   float(np.nanmean([r["au_pro"] for r in rows])))

    def as_dict(self) -> dict[str, float]:
        """Plain-dict form for the result payload + JSON artifact (stays serializable downstream)."""
        return asdict(self)

    def tracker_means(self) -> dict[str, float]:
        """The MLflow metric names for the aggregate (mean-over-categories) point estimates."""
        return {"img_auroc_mean": self.img_auroc, "au_pro_mean": self.au_pro}


@dataclass(frozen=True)
class EvalResult:
    """The eval result payload — what `Harness.aggregate`/`compute`/`run` return and every consumer
    (the report commands, the triad's gap report) reads. `ece` is `None` rather than absent when
    calibration didn't apply: the harness always sets it, so consumers can rely on it."""

    method: str
    per_category: list[CategoryRow]
    mean: Scores
    ci: dict[str, Bracket]
    ece: float | None
    per_defect: list[DefectRow]
    by_cat: list[ScoreArrays]

    def as_artifact(self) -> dict[str, object]:
        """JSON-artifact form: every field except `by_cat` (raw arrays go to the npz, not the json)."""
        return {"method": self.method, "per_category": self.per_category, "mean": self.mean.as_dict(),
                "ci": self.ci, "ece": self.ece, "per_defect": self.per_defect}


class Harness:
    """The single per-category eval spine — every method is a (fit_fn, score_fn) pair fed through it."""

    @staticmethod
    def aggregate(
        method: str, fit_fn: FitFn[StateT], score_fn: ScoreFn[StateT], cats: list[str]
    ) -> EvalResult:
        """Run the method over categories (per-category fit+score, ETA-logged), then compute the metrics."""
        prog = Progress(len(cats), tag=method)
        by_cat = []
        for c in cats:
            by_cat.append(ScoreArrays(*score_fn(fit_fn(c), c)))        # named fields — no fragile tuple order
            prog.tick(c)                                               # done/total + ETA (spot a slow run live)
        return Harness.compute(method, by_cat, cats)

    @staticmethod
    def from_artifact(npz: Mapping[str, np.ndarray]) -> EvalResult:
        """Recompute the full result OFFLINE from persisted per-image predictions (Predictions.pack npz) —
        no fit/score, no GPU. A new bootstrap / macro fix / ECE variant becomes a re-`compute`, not a re-run."""
        cats, by_cat = predictions.Predictions.unpack(npz)
        return Harness.compute(f"recompute[{len(cats)}cat]", by_cat, cats)

    @staticmethod
    def compute(method: str, by_cat: list[ScoreArrays], cats: list[str]) -> EvalResult:
        """The pure metric/CI core over per-category ScoreArrays — shared by live eval (`aggregate`) and
        offline recompute (`from_artifact`). Every headline number is a function of these arrays alone."""
        rows: list[CategoryRow] = []
        pool: dict[str, list[np.ndarray]] = {f: [] for f in ScoreArrays._fields}
        for c, sa in zip(cats, by_cat, strict=True):
            rows.append({"category": c, "n": len(sa.scores),
                         "img_auroc": metrics.Metrics.image_auroc(sa.scores, sa.labels),
                         "au_pro": metrics.Metrics.au_pro(sa.amaps, sa.masks, sa.valids)})
            log.info(f"  {c:12s}  img_auroc {rows[-1]['img_auroc']:.3f}   au_pro {rows[-1]['au_pro']:.3f}")
            for f in ScoreArrays._fields:
                pool[f].append(getattr(sa, f))

        mean = Scores.macro(rows)
        log.info(f"  {'MEAN':12s}  img_auroc {mean.img_auroc:.3f}   au_pro {mean.au_pro:.3f}")

        pooled = ScoreArrays(**{f: np.concatenate(pool[f]) for f in ScoreArrays._fields})
        defect_rows = cast("list[DefectRow]", diagnostics.Diagnostics.by_defect(pooled))
        log.info("  --- per defect type ---")
        for d in defect_rows:
            log.info(f"  {d['defect']:14s}  img_auroc {d['img_auroc']:.3f}   au_pro {d['au_pro']:.3f}  (n={d['n']})")

        # uncertainty from THIS run's test set — a stratified bootstrap over the images (not a retrain).
        # MACRO (mean-of-categories) to match the reported point: resample within each category, average.
        auroc_cat = [(sa.scores, sa.labels) for sa in by_cat]
        pro_cat = [(sa.amaps, sa.masks, sa.valids) for sa in by_cat]
        with Progress.stage("bootstrap CI"):                   # the au_pro bootstrap is the usual slow stage
            brackets: dict[str, Bracket] = {
                "img_auroc": metrics.Metrics.boot_macro_ci(metrics.Metrics.image_auroc, auroc_cat),
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

        result = EvalResult(method=method, per_category=rows, mean=mean, ci=brackets, ece=calib,
                            per_defect=defect_rows, by_cat=by_cat)
        invariants.Invariants.check(result.mean.as_dict(), result.ci, result.ece,
                                    result.by_cat)                # loud at run end if a number is inconsistent
        return result

    @staticmethod
    def log(result: EvalResult) -> None:  # pragma: no cover  mlflow IO; aggregate is the pure core
        rows = result.per_category
        tracking.Tracker.metrics(result.mean.tracker_means())
        for metric, (_point, low, high) in result.ci.items():   # bootstrap brackets next to the point metric
            if not np.isnan(low):                               # skip NaN (metric undefined on the run)
                tracking.Tracker.metrics({f"{metric}_lo": low, f"{metric}_hi": high})
        if result.ece is not None:
            tracking.Tracker.metrics({"ece": result.ece})
        tracking.Tracker.per_group("au_pro", {row["category"]: row["au_pro"] for row in rows})
        tracking.Tracker.per_group("img_auroc", {row["category"]: row["img_auroc"] for row in rows})
        tracking.Tracker.per_group("au_pro_defect",
                                   {row["defect"]: row["au_pro"] for row in result.per_defect})
        tracking.Tracker.artifact_json(                     # every EvalResult field but `by_cat` is JSON-native
            "aggregate.json", cast("dict[str, tracking.JsonValue]", result.as_artifact()))
        cats = [row["category"] for row in rows]                # persist per-image predictions -> offline recompute
        tracking.Tracker.artifact_npz("predictions.npz", predictions.Predictions.pack(result.by_cat, cats))

    @staticmethod
    def run(
        method: str,
        method_obj: AnomalyMethod[StateT] | Method[StateT],
        cats: list[str] | None = None,
        run_id: str | None = None,
        params: Mapping[str, tracking.ParamTree] | None = None,
    ) -> EvalResult:  # pragma: no cover  mlflow wrapper
        cats = cats or mvtec.Mvtec().categories()
        result = Harness.aggregate(method, method_obj.fit, method_obj.score, cats)
        if run_id:                                       # log into an existing run (e.g. the train run)
            with tracking.Tracker.resume(run_id):
                Harness.log(result)
        else:
            with tracking.Tracker.run("surfscan", method, params=params or RunParams(method, cats).as_dict()):
                Harness.log(result)
        return result
