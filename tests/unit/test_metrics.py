"""Unit tests for the eval metrics — the contribution, so they get verified.

Equivalence-class: AU-PRO on a perfect map -> ~1, random -> ~0.15 (the diagonal); image-AUROC on
perfect/reversed/single-class. No dataset needed.
"""
import numpy as np

from surfscan.evaluation.metrics import Metrics


def _masks(n=20, h=64, w=64):
    masks = np.zeros((n, h, w), bool)
    masks[:, 28:36, 28:36] = True          # one central defect region per frame
    valids = np.ones((n, h, w), bool)
    return masks, valids


def test_au_pro_perfect():
    rng = np.random.RandomState(0)
    masks, valids = _masks()
    perfect = masks.astype(float) + rng.rand(*masks.shape) * 1e-3
    assert Metrics.au_pro(perfect, masks, valids) > 0.97


def test_au_pro_random_is_near_diagonal():
    rng = np.random.RandomState(0)
    masks, valids = _masks()
    randm = rng.rand(*masks.shape)
    assert 0.05 < Metrics.au_pro(randm, masks, valids) < 0.30      # ~0.15, the diagonal


def test_image_auroc():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert Metrics.image_auroc(scores, labels) == 1.0             # perfect separation
    assert Metrics.image_auroc(-scores, labels) == 0.0            # reversed
    assert np.isnan(Metrics.image_auroc(scores, np.array([0, 0, 0, 0])))   # single class -> undefined


def test_ece_perfect_calibration():
    # probs == binary targets -> confidence matches accuracy in every bin -> ECE ~ 0
    rng = np.random.RandomState(0)
    valids = np.ones((5, 32, 32), bool)
    targets = rng.rand(5, 32, 32) < 0.3
    assert Metrics.ece(targets.astype(float), targets, valids) < 1e-6


def test_ece_overconfident_under_shift():
    # predict 0.95 everywhere but ~1% of pixels are actually defective -> big confidence-accuracy gap
    valids = np.ones((5, 32, 32), bool)
    targets = np.zeros((5, 32, 32), bool)
    targets[:, :3, :3] = True
    assert Metrics.ece(np.full((5, 32, 32), 0.95), targets, valids) > 0.8


def test_ece_calibrated_but_uncertain():
    # predict 0.5 everywhere with ~50% positives -> confidence == accuracy -> low ECE
    rng = np.random.RandomState(1)
    valids = np.ones((6, 32, 32), bool)
    targets = rng.rand(6, 32, 32) < 0.5
    assert Metrics.ece(np.full((6, 32, 32), 0.5), targets, valids) < 0.05


def test_metrics_nan_on_no_valid_pixels():
    # no valid pixels anywhere -> both metrics are undefined, not a crash
    a = np.zeros((2, 8, 8))
    empty = np.zeros((2, 8, 8), bool)
    assert np.isnan(Metrics.ece(a, empty, empty))
    assert np.isnan(Metrics.au_pro(a, empty, empty))
