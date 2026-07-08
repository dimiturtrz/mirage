"""Unit tests for the model forward passes — CPU dummies, small nets (no backbone download).

Equivalence class: each model consumes (N,C,H,W) and returns the documented shape(s). Smoke-level —
guards against a broken layer wiring / shape mismatch after a refactor.
"""
import numpy as np
import torch

from surfscan.models.draem import Draem, UNet, synthesize
from surfscan.models.inpaint import InpaintAE
from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import ModelCfg


def _cfg():
    return ModelCfg(in_ch=3, base=8, latent=32, size=64, depth=3)


def test_convvae_forward():
    x = torch.randn(2, 3, 64, 64)
    recon, mu, logvar = ConvVAE(_cfg())(x)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == (2, 32)


def test_inpaint_forward():
    x = torch.randn(2, 3, 64, 64)
    assert InpaintAE(_cfg())(x).shape == x.shape


def test_draem_forward():
    x = torch.randn(2, 3, 64, 64)
    rec, logits = Draem(3, base=8)(x)
    assert rec.shape == x.shape
    assert logits.shape == (2, 1, 64, 64)


def test_unet_forward():
    x = torch.randn(2, 3, 64, 64)
    assert UNet(3, 1, base=8)(x).shape == (2, 1, 64, 64)


def test_draem_synthesize_on_object():
    x = torch.rand(2, 3, 32, 32)
    valid = torch.zeros(2, 1, 32, 32)
    valid[:, :, 8:24, 8:24] = 1.0
    aug, mask = synthesize(x, valid, np.random.RandomState(0))
    assert aug.shape == x.shape and mask.shape == (2, 1, 32, 32)
    assert (mask.bool() & (valid < 0.5)).sum() == 0        # synthetic defect stays on-object
