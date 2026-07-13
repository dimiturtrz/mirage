"""Unit tests for the reconstruction-based anomaly maps — now device-agnostic, so they run on CPU.

A tiny model + fake data (N,C,H,W tensors on CPU) exercise the scoring path (autocast no-ops off cuda).
Equivalence class: output is (N,H,W), non-negative (squared error), background zeroed by the valid mask.
"""
import numpy as np
import polars as pl
import torch

from surfscan.evaluation.scoring import Scoring
from surfscan.models.inpaint import InpaintAE
from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import ModelCfg


class _Data:
    def __init__(self, x, valid):
        self.x = x
        self.valid = valid

    def __len__(self):
        return len(self.x)


def _cfg():
    return ModelCfg(in_ch=3, base=8, latent=16, size=32, depth=3)


def test_anomaly_maps_cpu():
    data = _Data(torch.randn(3, 3, 32, 32), torch.ones(3, 1, 32, 32))
    a = Scoring(ConvVAE(_cfg())).anomaly_maps(data, batch=2)
    assert a.shape == (3, 32, 32)
    assert (a >= 0).all()


def test_anomaly_maps_zeroed_outside_valid():
    valid = torch.zeros(2, 1, 32, 32)
    valid[:, :, 8:24, 8:24] = 1.0
    a = Scoring(ConvVAE(_cfg())).anomaly_maps(_Data(torch.randn(2, 3, 32, 32), valid), batch=2)
    assert a[:, 0, 0].sum() == 0                     # background corner -> zero


def test_inpaint_maps_cpu():
    data = _Data(torch.randn(3, 3, 32, 32), torch.ones(3, 1, 32, 32))
    a = Scoring(InpaintAE(_cfg())).inpaint_maps(data, grid=4, batch=2)
    assert a.shape == (3, 32, 32)
    assert (a >= 0).all()


def test_latents_shape_cpu():
    data = _Data(torch.randn(5, 3, 32, 32), torch.ones(5, 1, 32, 32))
    z = Scoring(ConvVAE(_cfg())).latents(data, batch=2)
    assert z.shape == (5, 16)                         # N x latent


def test_mahalanobis_far_scores_higher():
    rng = np.random.RandomState(0)
    train = rng.randn(200, 8)                         # normal latent cloud ~ N(0, I)
    test = np.stack([np.zeros(8), np.full(8, 6.0)])   # a near point + a far point
    d = Scoring.mahalanobis(train, test)
    assert d.shape == (2,)
    assert d[1] > d[0] and d[0] < 2.0                 # far off-cloud >> at the mean


class _Split:                                         # minimal stand-in for a GpuSplit
    def __init__(self, valid, gt, labels, defects):
        self.valid, self.gt = valid, gt
        self.df = pl.DataFrame({"label": labels, "defect": defects})


def test_score_arrays_assembles_tuple():
    n = 3
    valid = torch.ones(n, 1, 8, 8)
    gt = torch.zeros(n, 1, 8, 8); gt[1, 0, 2:4, 2:4] = 1
    amaps = np.random.RandomState(0).rand(n, 8, 8)
    a, v, m, s, labels, defects = Scoring.score_arrays(amaps, _Split(valid, gt, [0, 1, 0], ["good", "hole", "good"]))
    assert v.shape == (n, 8, 8) and v.all() and v.dtype == bool
    assert m[1].any() and not m[0].any()              # gt -> boolean mask, only sample 1 has a defect
    assert s.shape == (n,) and labels.tolist() == [0, 1, 0] and list(defects) == ["good", "hole", "good"]


def test_score_arrays_scores_override():
    split = _Split(torch.ones(2, 1, 4, 4), torch.zeros(2, 1, 4, 4), [0, 1], ["good", "hole"])
    custom = np.array([0.1, 0.9])
    _, _, _, s, _, _ = Scoring.score_arrays(np.zeros((2, 4, 4)), split, scores=custom)
    assert (s == custom).all()                        # passed-in scores are used verbatim
