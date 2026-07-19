"""Persisted predictions round-trip to identical metrics — so a CI/metric recompute needs no re-run.

The whole point of persisting per-image ScoreArrays: every headline number is a pure function of them,
so reloading the npz and recomputing must equal the live result exactly (bootstrap CIs use a fixed
default seed, so even the intervals match). Equivalence-class: perfect maps + a random arm.
"""
import numpy as np

from core.method import ScoreArrays
from surfscan.evaluation.harness import Harness
from surfscan.evaluation.predictions import Predictions


def _by_cat(kind="perfect", ncat=2, n=12, h=48, w=48):
    rng = np.random.RandomState(0)
    half = n // 2
    masks = np.zeros((n, h, w), bool)
    masks[half:, 20:28, 20:28] = True
    valids = np.ones((n, h, w), bool)
    labels = np.array([0] * half + [1] * half)
    defects = np.array(["good"] * half + ["hole"] * half)
    out = []
    for _ in range(ncat):
        amaps = masks.astype(float) + rng.rand(n, h, w) * 1e-3 if kind == "perfect" else rng.rand(n, h, w)
        scores = amaps.reshape(n, -1).max(1)
        out.append(ScoreArrays(amaps, valids, masks, scores, labels, defects))
    return out, [f"cat{i}" for i in range(ncat)]


def test_pack_unpack_exact():
    by_cat, cats = _by_cat()
    cats2, by_cat2 = Predictions.unpack(Predictions.pack(by_cat, cats))
    assert cats2 == cats
    for a, b in zip(by_cat, by_cat2, strict=True):
        for f in ScoreArrays._fields:
            assert np.array_equal(getattr(a, f), getattr(b, f))


def test_roundtrip_through_npz_file_gives_identical_metrics(tmp_path):
    by_cat, cats = _by_cat(kind="perfect")
    live = Harness.compute("m", by_cat, cats)

    p = tmp_path / "predictions.npz"                       # exercise the real np.savez/np.load path
    np.savez_compressed(p, **Predictions.pack(by_cat, cats))
    with np.load(p, allow_pickle=False) as d:
        recomputed = Harness.from_artifact(d)

    assert recomputed.mean == live.mean              # points identical
    assert recomputed.ci == live.ci                  # bootstrap CIs identical (fixed default seed)
    assert recomputed.ece == live.ece


def test_recompute_matches_for_random_arm(tmp_path):
    by_cat, cats = _by_cat(kind="random")
    live = Harness.compute("m", by_cat, cats)
    p = tmp_path / "p.npz"
    np.savez_compressed(p, **Predictions.pack(by_cat, cats))
    with np.load(p, allow_pickle=False) as d:
        assert Harness.from_artifact(d).ci == live.ci
