"""The anomaly-detection metrics — image AUROC + pixel AU-PRO.

AU-PRO (Per-Region Overlap) is *the* MVTec localization metric: average per-defect-region
overlap (TPR) integrated against the global false-positive rate up to a cap (0.3), normalized.
NOT pixel-AUROC — defects are tiny, so pixel-AUROC flatters. Image AUROC is detection.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label
from sklearn.metrics import roc_auc_score


def image_auroc(scores, labels):
    labels = np.asarray(labels)
    if labels.min() == labels.max():          # one class only -> undefined
        return float("nan")
    return float(roc_auc_score(labels, scores))


def ece(probs, targets, valids, n_bins=15):
    """Expected Calibration Error over valid object pixels. probs ∈ [0,1] (per-pixel defect
    probability), targets = binary defect mask, valids = object mask. Bins predictions and sums
    |accuracy − confidence| weighted by bin population. The honest 'is the score a probability?'
    number — under sim-to-real shift a synth-trained model is typically over-confident (ECE ↑).
    amaps/masks/valids: (N,H,W)."""
    p, t = [], []
    for pr, tg, v in zip(probs, targets, valids):
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


def au_pro(amaps, masks, valids, num_th=200, fpr_limit=0.3):
    """Per-region-overlap AUC up to fpr_limit. amaps/masks/valids: (N,H,W)."""
    region_vals, normal_vals = [], []
    gmin, gmax = np.inf, -np.inf
    for a, m, v in zip(amaps, masks, valids):
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
