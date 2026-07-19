"""Run-end invariants catch a plausible-but-wrong number — each check flags its seeded violation.

The historical bug: a gap CI computed on the pooled aggregate while the point was macro. `_ci_matches_mean`
and `reconciles` are the guards for exactly that; the rest guard brackets + ECE preconditions.
"""
import numpy as np
import pytest

from core.invariants import Invariants
from core.method import ScoreArrays


def _clean():
    return {"mean": {"img_auroc": 0.8, "au_pro": 0.7},
            "ci": {"img_auroc": (0.8, 0.75, 0.85), "au_pro": (0.7, 0.66, 0.74)},
            "ece": None, "by_cat": []}


def test_clean_result_has_no_violations():
    assert Invariants.check(**_clean()) == []


def test_bracket_excluding_point_flagged():
    res = _clean()
    res["ci"]["au_pro"] = (0.7, 0.72, 0.80)                 # point 0.70 below lo 0.72
    v = Invariants.check(**res)
    assert any("bracket" in m for m in v)


def test_ci_point_not_matching_mean_flagged():
    # the macro/micro bug shape: ci-point (pooled) disagrees with the reported (macro) mean
    res = _clean()
    res["ci"]["au_pro"] = (0.9, 0.86, 0.94)                 # ci-point 0.90 != mean 0.70
    v = Invariants.check(**res)
    assert any("!= reported mean" in m for m in v)


def test_ece_with_unbounded_amaps_flagged():
    res = _clean()
    res["ece"] = 0.05
    amaps = np.full((2, 4, 4), 5.0)                         # residual maps, NOT in [0,1]
    z = np.zeros((2, 4, 4), bool)
    res["by_cat"] = [ScoreArrays(amaps, z, z, np.zeros(2), np.zeros(2), np.array(["x", "x"]))]
    v = Invariants.check(**res)
    assert any("out of [0,1]" in m for m in v)


def test_strict_raises_on_violation():
    res = _clean()
    res["ci"]["au_pro"] = (0.9, 0.86, 0.94)
    with pytest.raises(AssertionError):
        Invariants.check(**res, strict=True)


def test_reconciles_true_when_delta_matches_and_false_otherwise():
    assert Invariants.reconciles(0.166, 0.628, 0.794, "gap") is True        # 0.794 - 0.628 == 0.166
    assert Invariants.reconciles(0.322, 0.628, 0.794, "gap") is False       # the pooled-vs-macro slip
