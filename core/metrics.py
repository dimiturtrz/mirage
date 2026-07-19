"""The shared metric / calibration / sim-to-real substrate — one home in `core`, scored by both legs.

Perception scores a `ScoreArrays` here (image AUROC + pixel AU-PRO + calibration ECE); control scores a
`Trajectory`'s per-episode success through the *same* paired bootstrap (`boot_delta_ci`) that brackets the
perception sim-to-real gap. The primitives are pure numpy/sklearn/skimage — no leg-specific type crosses this
boundary — so `core` stays the independent kernel and the two legs share one honest uncertainty story instead
of each rolling its own.

AU-PRO (Per-Region Overlap) is *the* MVTec localization metric: average per-defect-region
overlap (TPR) integrated against the global false-positive rate up to a cap (0.3), normalized.
NOT pixel-AUROC — defects are tiny, so pixel-AUROC flatters. Image AUROC is detection. The bootstrap CIs
(`boot_ci` / `boot_delta_ci` / the macro variants) resample the evaluation set — a test-set N, not a retrain —
so one run yields `point [lo, hi]`; the paired delta cancels shared noise to answer "does B differ from A?".
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, Float, Int, Shaped
from skimage.measure import label
from sklearn.metrics import roc_auc_score

MetricValue: TypeAlias = float | np.float64
MetricFn: TypeAlias = Callable[..., MetricValue]


@dataclass(frozen=True)
class BootCfg:
    """Bootstrap knobs: resample count, two-sided level, and the resampling RNG (a bootstrap seed,
    unrelated to any training seed). `rng=None` -> a fixed default_rng(0) so intervals are reproducible."""

    n_boot: int = 1000
    alpha: float = 0.05
    rng: np.random.Generator | None = None

    def gen(self) -> np.random.Generator:
        return np.random.default_rng(0) if self.rng is None else self.rng


_DEFAULT_BOOT = BootCfg()   # frozen -> a safe shared default (no per-call mutation)


class Metrics:
    """The anomaly-detection metrics — image AUROC + pixel AU-PRO + calibration ECE.

    Uncertainty is a bootstrap over the test set, NOT a retrain: `boot_ci` resamples the N test
    images (with replacement), recomputes the metric per resample, and reports the percentile
    interval — power comes from the test-set N, so one run yields `point [lo, hi]`. `boot_delta_ci`
    is the PAIRED version for A-vs-B: it applies the SAME resampled image indices to both runs, so
    the shared test-set noise cancels and the interval is on the delta (does B beat A, honestly?)."""

    @staticmethod
    def _resample(
        arrays: Sequence[Shaped[np.ndarray, "n *rest"]], idx: Int[np.ndarray, "n"]
    ) -> list[Shaped[np.ndarray, "n *rest"]]:
        """Index the first axis (= image) of every metric-input array by one bootstrap draw."""
        return [np.asarray(a)[idx] for a in arrays]

    @staticmethod
    def _interval(
        point: float, samples: Sequence[MetricValue], alpha: float
    ) -> tuple[float, float, float]:
        """(point, lo, hi) — percentile interval over the finite bootstrap samples (NaNs dropped)."""
        s = np.asarray(samples, dtype=np.float64)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return (point, float("nan"), float("nan"))
        lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (point, float(lo), float(hi))

    @staticmethod
    def boot_ci(
        metric_fn: MetricFn, *arrays: Shaped[np.ndarray, "n *rest"], cfg: BootCfg = _DEFAULT_BOOT
    ) -> tuple[float, float, float]:
        """(point, lo, hi) for `metric_fn(*arrays)` — percentile bootstrap over the test images.

        `metric_fn` takes the arrays in order (e.g. image_auroc over (scores, labels), au_pro over
        (amaps, masks, valids)); every array's first axis is the image, resampled by a shared index."""
        arrays_list = [np.asarray(a) for a in arrays]
        n = len(arrays_list[0])
        point = float(metric_fn(*arrays_list))
        draws = cfg.gen().integers(0, n, size=(cfg.n_boot, n))
        samples = [metric_fn(*Metrics._resample(arrays_list, idx)) for idx in draws]
        return Metrics._interval(point, samples, cfg.alpha)

    @staticmethod
    def boot_delta_ci(
        metric_fn: MetricFn,
        arrays_a: Sequence[Shaped[np.ndarray, "n *rest"]],
        arrays_b: Sequence[Shaped[np.ndarray, "n *rest"]],
        cfg: BootCfg = _DEFAULT_BOOT,
    ) -> tuple[float, float, float]:
        """(delta, lo, hi) for metric(B) − metric(A), PAIRED bootstrap. Same resampled image indices
        hit both runs each draw, so shared test-set noise cancels — a CI that excludes 0 is an honest
        'B differs from A'. Requires A and B scored on the SAME test images in the SAME order."""
        arrays_a_list = [np.asarray(a) for a in arrays_a]
        arrays_b_list = [np.asarray(a) for a in arrays_b]
        n = len(arrays_a_list[0])
        delta = float(metric_fn(*arrays_b_list)) - float(metric_fn(*arrays_a_list))
        draws = cfg.gen().integers(0, n, size=(cfg.n_boot, n))
        samples = [metric_fn(*Metrics._resample(arrays_b_list, idx))
                   - metric_fn(*Metrics._resample(arrays_a_list, idx))
                   for idx in draws]
        return Metrics._interval(delta, samples, cfg.alpha)

    # --- macro (mean-of-categories) bootstrap: the CI matches the reported statistic. A pooled/micro
    # resample would over-weight image-dense categories; MVTec reports the per-category mean, so the
    # bootstrap resamples images WITHIN each category and averages per draw (stratified). ---

    @staticmethod
    def _nanmean(values: Sequence[MetricValue]) -> float:
        """Mean over finite values -> NaN if none are finite (no 'empty slice' warning for a dead draw)."""
        finite = [value for value in values if not np.isnan(value)]
        return float(np.mean(finite)) if finite else float("nan")

    @staticmethod
    def _macro_point(metric_fn: MetricFn, per_cat: Sequence[Sequence[Shaped[np.ndarray, "n *rest"]]]) -> float:
        """mean-of-categories of `metric_fn` over each category's full arg-tuple (NaN categories dropped)."""
        return Metrics._nanmean([metric_fn(*args) for args in per_cat])

    @staticmethod
    def _macro_draw(
        metric_fn: MetricFn,
        per_cat: Sequence[Sequence[Shaped[np.ndarray, "n *rest"]]],
        idxs: Sequence[Int[np.ndarray, "n"]],
    ) -> float:
        """One stratified draw: resample within each category by its index array, then mean-of-categories."""
        return Metrics._nanmean([metric_fn(*Metrics._resample(a, ix)) for a, ix in zip(per_cat, idxs, strict=True)])

    @staticmethod
    def boot_macro_ci(
        metric_fn: MetricFn,
        per_cat: Sequence[Sequence[Shaped[np.ndarray, "n *rest"]]],
        cfg: BootCfg = _DEFAULT_BOOT,
    ) -> tuple[float, float, float]:
        """(point, lo, hi) for the mean-of-categories metric — stratified bootstrap (resample images within
        each category, average). `per_cat` = one metric arg-tuple per category. The CI for a macro number."""
        rng = cfg.gen()
        point = Metrics._macro_point(metric_fn, per_cat)
        samples = [Metrics._macro_draw(metric_fn, per_cat,
                                       [rng.integers(0, len(a[0]), len(a[0])) for a in per_cat])
                   for _ in range(cfg.n_boot)]
        return Metrics._interval(point, samples, cfg.alpha)

    @staticmethod
    def boot_macro_delta_ci(
        metric_fn: MetricFn,
        per_cat_a: Sequence[Sequence[Shaped[np.ndarray, "n *rest"]]],
        per_cat_b: Sequence[Sequence[Shaped[np.ndarray, "n *rest"]]],
        cfg: BootCfg = _DEFAULT_BOOT,
    ) -> tuple[float, float, float]:
        """(delta, lo, hi) for macro-metric(B) − macro-metric(A), PAIRED + stratified. Same within-category
        resample hits both arms each draw (shared eval images per category, same order), so the CI is on the
        mean-of-categories delta — the headline gap number, honestly bracketed."""
        rng = cfg.gen()
        delta = Metrics._macro_point(metric_fn, per_cat_b) - Metrics._macro_point(metric_fn, per_cat_a)
        samples = []
        for _ in range(cfg.n_boot):
            idxs = [rng.integers(0, len(a[0]), len(a[0])) for a in per_cat_a]     # paired: same idx to A and B
            samples.append(Metrics._macro_draw(metric_fn, per_cat_b, idxs)
                           - Metrics._macro_draw(metric_fn, per_cat_a, idxs))
        return Metrics._interval(delta, samples, cfg.alpha)

    @staticmethod
    def image_auroc(
        scores: Float[np.ndarray, "n"], labels: Int[np.ndarray, "n"]
    ) -> float:
        labels = np.asarray(labels)
        if labels.min() == labels.max():          # one class only -> undefined
            return float("nan")
        return float(roc_auc_score(labels, scores))

    @staticmethod
    def ece(
        probs: Float[np.ndarray, "n h w"],
        targets: Bool[np.ndarray, "n h w"],
        valids: Bool[np.ndarray, "n h w"],
        n_bins: int = 15,
    ) -> float:
        """Expected Calibration Error over valid object pixels. probs ∈ [0,1] (per-pixel defect
        probability), targets = binary defect mask, valids = object mask. Bins predictions and sums
        |accuracy − confidence| weighted by bin population. The measured 'is the score a probability?'
        number — under sim-to-real shift a synth-trained model is typically over-confident (ECE ↑).
        amaps/masks/valids: (N,H,W)."""
        p_parts: list[np.ndarray] = []
        t_parts: list[np.ndarray] = []
        for pr, tg, v in zip(probs, targets, valids, strict=True):
            vb = v.astype(bool)
            p_parts.append(np.asarray(pr)[vb].ravel())
            t_parts.append(np.asarray(tg)[vb].astype(np.float64).ravel())
        p = np.concatenate(p_parts) if p_parts else np.array([])
        t = np.concatenate(t_parts) if t_parts else np.array([])
        if p.size == 0:
            return float("nan")
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        e = 0.0
        for b in range(n_bins):
            m = idx == b
            if not m.any():
                continue
            e += (m.sum() / p.size) * abs(t[m].mean() - p[m].mean())   # |acc − conf|
        return float(e)

    @staticmethod
    def au_pro(
        amaps: Float[np.ndarray, "n h w"],
        masks: Bool[np.ndarray, "n h w"],
        valids: Bool[np.ndarray, "n h w"],
        num_th: int = 200,
        fpr_limit: float = 0.3,
    ) -> float:
        """Per-region-overlap AUC up to fpr_limit. amaps/masks/valids: (N,H,W)."""
        region_vals: list[np.ndarray] = []
        normal_vals: list[np.ndarray] = []
        gmin, gmax = np.inf, -np.inf
        for a, m, v in zip(amaps, masks, valids, strict=True):
            m = m.astype(bool); v = v.astype(bool)
            if m.any():
                lab = label(m)
                for r in range(1, lab.max() + 1):
                    rv = a[lab == r]
                    region_vals.append(rv)
                    gmin, gmax = min(gmin, rv.min()), max(gmax, rv.max())
            nm = v & ~m                            # normal = valid object pixels that aren't defect
            if nm.any():
                nv = a[nm]
                normal_vals.append(nv)
                gmin, gmax = min(gmin, nv.min()), max(gmax, nv.max())
        if not region_vals or not normal_vals:
            return float("nan")

        normal = np.sort(np.concatenate(normal_vals))
        ths = np.linspace(gmin, gmax, num_th)
        fpr = 1.0 - np.searchsorted(normal, ths, side="right") / len(normal)
        pros = np.zeros(num_th)
        for rv in region_vals:                     # per-region TPR(th), averaged over all regions
            rs = np.sort(rv)
            pros += 1.0 - np.searchsorted(rs, ths, side="right") / len(rs)
        pros /= len(region_vals)

        # PRO curve = upper envelope of PRO per unique FPR (the curve is multivalued at a given FPR),
        # then integrate against a regular FPR grid up to the cap.
        uniq, inv = np.unique(fpr, return_inverse=True)
        env = np.zeros_like(uniq)
        np.maximum.at(env, inv, pros)
        grid = np.linspace(0.0, fpr_limit, 256)
        pg = np.interp(grid, uniq, env)            # below min FPR -> env[0]; above -> env[-1]
        return float(np.trapezoid(pg, grid) / fpr_limit)   # np.trapz renamed in numpy 2.x
