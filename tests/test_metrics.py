"""Unit tests for the eval metrics — the contribution, so they get verified.

Equivalence-class: AU-PRO on a perfect map -> ~1, random -> ~0.15 (the diagonal); image-AUROC on
perfect/reversed/single-class. No dataset needed.
"""
import numpy as np

from mirage.evaluation.metrics import au_pro, image_auroc


def _masks(n=20, h=64, w=64):
    masks = np.zeros((n, h, w), bool)
    masks[:, 28:36, 28:36] = True          # one central defect region per frame
    valids = np.ones((n, h, w), bool)
    return masks, valids


def test_au_pro_perfect():
    rng = np.random.RandomState(0)
    masks, valids = _masks()
    perfect = masks.astype(float) + rng.rand(*masks.shape) * 1e-3
    assert au_pro(perfect, masks, valids) > 0.97


def test_au_pro_random_is_near_diagonal():
    rng = np.random.RandomState(0)
    masks, valids = _masks()
    randm = rng.rand(*masks.shape)
    assert 0.05 < au_pro(randm, masks, valids) < 0.30      # ~0.15, the diagonal


def test_image_auroc():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert image_auroc(scores, labels) == 1.0             # perfect separation
    assert image_auroc(-scores, labels) == 0.0            # reversed
    assert np.isnan(image_auroc(scores, np.array([0, 0, 0, 0])))   # single class -> undefined
