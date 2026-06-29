"""Unit test for image-level scoring — top-k mean of pixel residuals over valid pixels. No dataset."""
import numpy as np

from surfscan.evaluation.scoring import image_scores


def test_topk_mean_over_valid():
    a = np.zeros((1, 10, 10))
    a[0, 0, :5] = 4.0                                  # exactly k=5 high pixels
    v = np.ones((1, 10, 10), bool)
    assert abs(image_scores(a, v, k=5)[0] - 4.0) < 1e-9


def test_empty_valid_is_zero():
    a = np.ones((1, 4, 4))
    v = np.zeros((1, 4, 4), bool)                      # no valid pixels -> score 0, no crash
    assert image_scores(a, v)[0] == 0.0


def test_ignores_invalid_pixels():
    a = np.zeros((1, 8, 8))
    a[0, 0, 0] = 99.0                                  # a huge residual, but outside the object
    v = np.ones((1, 8, 8), bool)
    v[0, 0, 0] = False
    assert image_scores(a, v)[0] == 0.0                # masked out -> doesn't leak into the score
