"""Unit tests for the reconstruction-based anomaly maps — now device-agnostic, so they run on CPU.

A tiny model + fake data (N,C,H,W tensors on CPU) exercise the scoring path (autocast no-ops off cuda).
Equivalence class: output is (N,H,W), non-negative (squared error), background zeroed by the valid mask.
"""
import torch

from surfscan.evaluation.scoring import anomaly_maps, inpaint_maps
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
    a = anomaly_maps(ConvVAE(_cfg()), data, batch=2)
    assert a.shape == (3, 32, 32)
    assert (a >= 0).all()


def test_anomaly_maps_zeroed_outside_valid():
    valid = torch.zeros(2, 1, 32, 32)
    valid[:, :, 8:24, 8:24] = 1.0
    a = anomaly_maps(ConvVAE(_cfg()), _Data(torch.randn(2, 3, 32, 32), valid), batch=2)
    assert a[:, 0, 0].sum() == 0                     # background corner -> zero


def test_inpaint_maps_cpu():
    data = _Data(torch.randn(3, 3, 32, 32), torch.ones(3, 1, 32, 32))
    a = inpaint_maps(InpaintAE(_cfg()), data, grid=4, batch=2)
    assert a.shape == (3, 32, 32)
    assert (a >= 0).all()
