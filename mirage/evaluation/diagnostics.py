"""Per-condition (defect-type) stratified diagnostics — the eval-rigor breakdown.

Beyond the per-category mean: how well does the detector localize each DEFECT TYPE (hole / crack /
contamination / combined / ...)? Stratified failure, the way systole stratified EF by pathology —
the per-condition diagnostics the plan calls the contribution.

`by_defect` pools test samples across categories and stratifies by defect type: image-AUROC of
that type's anomalies vs all normals, and AU-PRO over that type's anomaly frames.
"""
from __future__ import annotations

import numpy as np

from mirage.evaluation import metrics


def by_defect(amaps, scores, masks, valids, labels, defects):
    labels = np.asarray(labels)
    defects = np.asarray(defects)
    types = sorted({d for d, l in zip(defects, labels) if l == 1})
    rows = []
    for t in types:
        is_t = (defects == t) & (labels == 1)
        # image-AUROC needs both classes -> this type's anomalies + ALL normals
        sel = is_t | (labels == 0)
        rows.append({
            "defect": t, "n": int(is_t.sum()),
            "img_auroc": metrics.image_auroc(scores[sel], labels[sel]),
            "au_pro": metrics.au_pro(amaps[is_t], masks[is_t], valids[is_t]),
        })
    return rows
