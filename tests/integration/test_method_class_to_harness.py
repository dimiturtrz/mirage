"""Integration: a configured AnomalyMethod CLASS flows through build_method -> harness.aggregate.

The f7w refactor turned methods from local (fit, score) closures into configured classes. This pins
that seam end-to-end without a backbone: a fake method class, constructed the way the runner does
(build_method: config -> instance), must satisfy the harness contract and produce a valid, localizing
aggregate. Synthetic maps, no dataset, no GPU, no torchvision.
"""
import argparse
from dataclasses import dataclass, fields

import numpy as np

from surfscan.evaluation.harness import Harness
from surfscan.method_cli import MethodCli


@dataclass(frozen=True)
class _Cfg:
    strength: float = 1.0


class _Method:
    """A minimal AnomalyMethod: score returns a near-perfect map (defect region lit) so the harness
    math is exercised on a signal it should score high — proves the class integrates, not the score."""
    run_name = "fake_class"

    def __init__(self, cfg, dev):
        self.cfg = cfg
        self.dev = dev

    def fit(self, cat):
        return {"cat": cat}                              # any state object

    def score(self, _state, _cat):
        n, h, w = 8, 32, 32
        rng = np.random.RandomState(0)
        masks = np.zeros((n, h, w), bool)
        masks[4:, 12:20, 12:20] = True                  # last 4 frames carry a defect region
        valids = np.ones((n, h, w), bool)
        amaps = masks.astype(float) * self.cfg.strength + rng.rand(n, h, w) * 1e-3
        scores = amaps.reshape(n, -1).max(1)
        labels = np.array([0] * 4 + [1] * 4)
        defects = np.array(["good"] * 4 + ["hole"] * 4)
        return amaps, valids, masks, scores, labels, defects


def test_method_class_flows_through_harness():
    args = argparse.Namespace(strength=1.0, cats=None)
    run_name, method = MethodCli.build_method(_Method, _Cfg, args, "cpu")   # the runner's construction path
    assert run_name == "fake_class" and method.dev == "cpu"

    res = Harness.aggregate(run_name, method.fit, method.score, ["a", "b"])
    assert {f.name for f in fields(res)} == {"method", "per_category", "mean", "ci", "ece",
                                        "per_defect", "by_cat"}
    assert len(res.per_category) == 2
    assert res.mean.au_pro > 0.9                  # class-form method integrates + localizes
    hole = next(d for d in res.per_defect if d["defect"] == "hole")
    assert hole["au_pro"] > 0.9
