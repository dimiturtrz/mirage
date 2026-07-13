"""The anomaly-detection metrics — image AUROC + pixel AU-PRO.

AU-PRO (Per-Region Overlap) is *the* MVTec localization metric: average per-defect-region
overlap (TPR) integrated against the global false-positive rate up to a cap (0.3), normalized.
NOT pixel-AUROC — defects are tiny, so pixel-AUROC flatters. Image AUROC is detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BootCfg:
    """Bootstrap knobs: resample count, two-sided level, and the resampling RNG (a bootstrap seed,
    unrelated to any training seed). `rng=None` -> a fixed default_rng(0) so intervals are reproducible."""

    n_boot: int = 1000
    alpha: float = 0.05
    rng: object = None

    def gen(self):
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
    def _resample(arrays, idx):
        """Index the first axis (= image) of every metric-input array by one bootstrap draw."""
        return [np.asarray(a)[idx] for a in arrays]

    @staticmethod
    def _interval(point, samples, alpha):
        """(point, lo, hi) — percentile interval over the finite bootstrap samples (NaNs dropped)."""
        s = np.asarray(samples, dtype=np.float64)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return (point, float("nan"), float("nan"))
        lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (point, float(lo), float(hi))

    @staticmethod
    def boot_ci(metric_fn, *arrays, cfg=_DEFAULT_BOOT):
        """(point, lo, hi) for `metric_fn(*arrays)` — percentile bootstrap over the test images.

        `metric_fn` takes the arrays in order (e.g. image_auroc over (scores, labels), au_pro over
        (amaps, masks, valids)); every array's first axis is the image, resampled by a shared index."""
        arrays = [np.asarray(a) for a in arrays]
        n = len(arrays[0])
        point = float(metric_fn(*arrays))
        draws = cfg.gen().integers(0, n, size=(cfg.n_boot, n))
        samples = [metric_fn(*Metrics._resample(arrays, idx)) for idx in draws]
        return Metrics._interval(point, samples, cfg.alpha)

    @staticmethod
    def boot_delta_ci(metric_fn, arrays_a, arrays_b, cfg=_DEFAULT_BOOT):
        """(delta, lo, hi) for metric(B) − metric(A), PAIRED bootstrap. Same resampled image indices
        hit both runs each draw, so shared test-set noise cancels — a CI that excludes 0 is an honest
        'B differs from A'. Requires A and B scored on the SAME test images in the SAME order."""
        arrays_a = [np.asarray(a) for a in arrays_a]
        arrays_b = [np.asarray(a) for a in arrays_b]
        n = len(arrays_a[0])
        delta = float(metric_fn(*arrays_b)) - float(metric_fn(*arrays_a))
        draws = cfg.gen().integers(0, n, size=(cfg.n_boot, n))
        samples = [metric_fn(*Metrics._resample(arrays_b, idx)) - metric_fn(*Metrics._resample(arrays_a, idx))
                   for idx in draws]
        return Metrics._interval(delta, samples, cfg.alpha)

    @staticmethod
    def image_auroc(scores, labels):
        labels = np.asarray(labels)
        if labels.min() == labels.max():          # one class only -> undefined
            return float("nan")
        return float(roc_auc_score(labels, scores))

    @staticmethod
    def ece(probs, targets, valids, n_bins=15):
        """Expected Calibration Error over valid object pixels. probs ∈ [0,1] (per-pixel defect
        probability), targets = binary defect mask, valids = object mask. Bins predictions and sums
        |accuracy − confidence| weighted by bin population. The measured 'is the score a probability?'
        number — under sim-to-real shift a synth-trained model is typically over-confident (ECE ↑).
        amaps/masks/valids: (N,H,W)."""
        p, t = [], []
        for pr, tg, v in zip(probs, targets, valids, strict=True):
            vb = v.astype(bool)
            p.append(np.asarray(pr)[vb].ravel())
            t.append(np.asarray(tg)[vb].astype(np.float64).ravel())
        p = np.concatenate(p) if p else np.array([])
        t = np.concatenate(t) if t else np.array([])
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
    def au_pro(amaps, masks, valids, num_th=200, fpr_limit=0.3):
        """Per-region-overlap AUC up to fpr_limit. amaps/masks/valids: (N,H,W)."""
        region_vals, normal_vals = [], []
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
