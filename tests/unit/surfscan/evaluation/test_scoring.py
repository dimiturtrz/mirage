"""Unit test for image-level scoring — top-k mean of pixel residuals over valid pixels. No dataset."""
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

from surfscan.evaluation.scoring import Scoring


def test_topk_mean_over_valid():
    a = np.zeros((1, 10, 10))
    a[0, 0, :5] = 4.0                                  # exactly k=5 high pixels
    v = np.ones((1, 10, 10), bool)
    assert abs(Scoring.image_scores(a, v, k=5)[0] - 4.0) < 1e-9


def test_empty_valid_is_zero():
    a = np.ones((1, 4, 4))
    v = np.zeros((1, 4, 4), bool)                      # no valid pixels -> score 0, no crash
    assert Scoring.image_scores(a, v)[0] == 0.0


def test_ignores_invalid_pixels():
    a = np.zeros((1, 8, 8))
    a[0, 0, 0] = 99.0                                  # a huge residual, but outside the object
    v = np.ones((1, 8, 8), bool)
    v[0, 0, 0] = False
    assert Scoring.image_scores(a, v)[0] == 0.0


def _split(n, h=4, w=4):
    """A minimal GpuSplit-like double for score_logits/score_arrays: (n,1,h,w) valid/gt + a label df."""
    valid = torch.ones(n, 1, h, w)
    gt = torch.zeros(n, 1, h, w)
    df = pl.DataFrame({"label": list(range(n)), "defect": ["good"] * n})
    return SimpleNamespace(valid=valid, gt=gt, df=df)


def test_score_logits_threads_forward_through_amap_tail():
    """DRAEM-style whole-split scoring: amaps = sigmoid(logits)*valid, one row per split row, batched."""
    n, h, w = 5, 4, 4
    test = _split(n, h, w)
    logits = torch.randn(n, 1, h, w)

    def forward(i0, i1):
        return logits[i0:i1]
    amaps, valids, _, _, labels, _ = Scoring.score_logits(forward, test.valid, test, batch=2)
    assert amaps.shape == (n, h, w)                           # n derived from valid, not passed
    assert np.allclose(amaps, torch.sigmoid(logits).squeeze(1).numpy())   # sigmoid amap over valid=1
    assert list(labels) == list(range(n))


def test_score_logits_idx_subset_selects_rows():
    """Triad-style: forward runs over a pre-subset row space; idx selects the split's label/mask rows."""
    n, h, w = 6, 4, 4
    test = _split(n, h, w)
    idx = np.array([1, 3, 4])                                 # the eval-half subset
    sub = torch.randn(len(idx), 1, h, w)

    def forward(i0, i1):
        return sub[i0:i1]
    amaps, _, _, _, labels, _ = Scoring.score_logits(forward, test.valid[idx], test, batch=2, idx=idx)
    assert amaps.shape == (len(idx), h, w)     # scored only the subset
    assert list(labels) == [1, 3, 4]           # labels from idx rows
