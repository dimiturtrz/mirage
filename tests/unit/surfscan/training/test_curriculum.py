"""Unit tests for the closed-loop curriculum — the differentiator, so it gets verified.

Equivalence-class: unseen kinds explore uniformly; a high-loss (weak) kind gets up-weighted; sample()
returns a single known kind and weights normalize. No dataset, no GPU.
"""
import numpy as np

from surfscan.training.curriculum import KindCurriculum

KINDS = ("dent", "scratch", "contamination", "bump")


def test_unseen_kinds_explore_uniformly():
    w = KindCurriculum(KINDS, seed=0)._weights()
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w, w[0])                       # no observations yet -> uniform


def test_weak_kind_upweighted():
    c = KindCurriculum(KINDS, seed=0)
    for _ in range(20):
        c.observe(("dent",), 2.0)                     # 'dent' = high loss = weak
        for k in ("scratch", "contamination", "bump"):
            c.observe((k,), 0.1)
    w = dict(zip(KINDS, c._weights(), strict=True))
    assert w["dent"] > w["scratch"]                   # detector's weak kind sampled more
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_sample_returns_single_known_kind():
    c = KindCurriculum(KINDS, seed=3)
    k = c.sample(16)
    assert isinstance(k, tuple) and len(k) == 1 and k[0] in KINDS


def test_observe_ema_tracks_loss():
    c = KindCurriculum(KINDS, seed=0, alpha=0.5)
    c.observe(("dent",), 1.0)
    c.observe(("dent",), 0.0)
    assert abs(c.loss["dent"] - 0.5) < 1e-9           # EMA: 0.5*1 + 0.5*0
