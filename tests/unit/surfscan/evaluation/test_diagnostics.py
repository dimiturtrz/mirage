"""Per-defect-type stratification — each anomaly type is scored vs ALL normals, one row per type.

Equivalence-class: two defect types + normals; by_defect emits a row per type with the right count,
and image-AUROC uses that type's anomalies against every normal (perfect scores -> ~1.0).
"""
import numpy as np

from core.method import ScoreArrays
from surfscan.evaluation.diagnostics import Diagnostics


def _sa(h=16, w=16):
    # 2 normals, 2 "hole", 2 "crack"; scores rank anomalies above normals (perfect separation)
    labels = np.array([0, 0, 1, 1, 1, 1])
    defects = np.array(["good", "good", "hole", "hole", "crack", "crack"])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.85, 0.95])
    masks = np.zeros((6, h, w), bool)
    masks[2:, 4:8, 4:8] = True
    valids = np.ones((6, h, w), bool)
    amaps = masks.astype(float) + 1e-3
    return ScoreArrays(amaps, valids, masks, scores, labels, defects)


def test_by_defect_one_row_per_type_with_counts():
    rows = Diagnostics.by_defect(_sa())
    by = {r["defect"]: r for r in rows}
    assert set(by) == {"hole", "crack"}                  # only anomaly types, normals excluded as a type
    assert by["hole"]["n"] == 2 and by["crack"]["n"] == 2
    assert by["hole"]["img_auroc"] == 1.0                # this type's anomalies rank above all normals
    assert by["crack"]["au_pro"] > 0.9                   # perfect maps localize the type


def test_by_defect_empty_when_no_anomalies():
    rng = np.random.RandomState(0)
    sa = ScoreArrays(rng.rand(3, 8, 8), np.ones((3, 8, 8), bool), np.zeros((3, 8, 8), bool),
                     rng.rand(3), np.zeros(3, int), np.array(["good", "good", "good"]))
    assert Diagnostics.by_defect(sa) == []               # no positive labels -> no defect-type rows
