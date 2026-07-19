"""Unit tests for the eval spine — harness.aggregate over a synthetic (fit, score) method, plus the
result carriers it produces (`Scores`, `EvalResult`).

aggregate is pure (no MLflow, no dataset), so the eval math is verified directly: perfect maps ->
au_pro≈1 and per-defect stratification present; random -> near the diagonal. Equivalence-class:
Scores round-trips to the payload dict and to the MLflow `_mean` metric names; `as_artifact` carries
every field but the raw arrays.
"""
import numpy as np

from surfscan.evaluation.harness import EvalResult, Harness, Scores


def _method(kind, n=12, h=48, w=48):
    """Two categories of n frames: half good (no gt), half a 'hole' defect with a central region."""
    rng = np.random.RandomState(0)
    half = n // 2
    masks = np.zeros((n, h, w), bool)
    masks[half:, 20:28, 20:28] = True                 # only the anomalous frames carry a gt region
    valids = np.ones((n, h, w), bool)
    labels = np.array([0] * half + [1] * half)
    defects = np.array(["good"] * half + ["hole"] * half)

    def fit(_c):
        return None

    def score(_state, _c):
        amaps = (masks.astype(float) + rng.rand(n, h, w) * 1e-3) if kind == "perfect" else rng.rand(n, h, w)
        scores = amaps.reshape(n, -1).max(1)
        return amaps, valids, masks, scores, labels, defects

    return ["a", "b"], fit, score


def test_scores_as_dict():
    assert Scores(0.9, 0.8).as_dict() == {"img_auroc": 0.9, "au_pro": 0.8}


def test_scores_tracker_means():
    assert Scores(0.9, 0.8).tracker_means() == {"img_auroc_mean": 0.9, "au_pro_mean": 0.8}


def test_scores_reconstruct_from_payload():
    s = Scores(0.42, 0.37)
    assert Scores(**s.as_dict()) == s


def test_eval_result_as_artifact_drops_arrays_and_flattens_mean():
    r = EvalResult(method="m", per_category=[], mean=Scores(0.9, 0.8), ci={"au_pro": (0.8, 0.7, 0.9)},
                   ece=None, per_defect=[], by_cat=[])
    art = r.as_artifact()
    assert set(art) == {"method", "per_category", "mean", "ci", "ece", "per_defect"}   # no raw arrays
    assert art["mean"] == {"img_auroc": 0.9, "au_pro": 0.8}                            # plain dict, JSON-native


def test_aggregate_structure_and_perfect():
    cats, fit, score = _method("perfect")
    res = Harness.aggregate("m", fit, score, cats)
    assert len(res.per_category) == 2
    assert res.mean.au_pro > 0.95
    assert res.mean.img_auroc > 0.95
    hole = next(d for d in res.per_defect if d["defect"] == "hole")
    assert hole["au_pro"] > 0.95                       # the defect type localizes
    for metric in ("img_auroc", "au_pro"):             # bootstrap CI present + brackets the point
        point, lo, hi = res.ci[metric]
        assert lo <= point <= hi


def test_aggregate_random_near_diagonal():
    cats, fit, score = _method("random")
    res = Harness.aggregate("m", fit, score, cats)
    assert 0.05 < res.mean.au_pro < 0.35               # ~diagonal, not perfect
    assert res.ece is not None and 0.0 <= res.ece <= 1.0   # amaps ∈ [0,1) -> ECE computed
