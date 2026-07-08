"""Unit tests for the VAE losses — pure math, tiny tensors.

Equivalence classes: perfect reconstruction -> 0; masked region ignored; standard-normal KL = 0.
"""
import torch

from surfscan.training.losses import kl_loss, masked_recon_loss


def test_masked_recon_perfect_is_zero():
    x = torch.randn(2, 3, 8, 8)
    valid = torch.ones(2, 1, 8, 8)
    assert masked_recon_loss(x.clone(), x, valid).item() < 1e-6


def test_masked_recon_ignores_background():
    x = torch.zeros(1, 1, 4, 4)
    recon = x.clone()
    recon[0, 0, 0, 0] = 5.0                       # a big error, but on a masked-out pixel
    valid = torch.ones(1, 1, 4, 4)
    valid[0, 0, 0, 0] = 0.0                        # that pixel is background
    assert masked_recon_loss(recon, x, valid).item() < 1e-6


def test_masked_recon_known_value():
    x = torch.zeros(1, 1, 2, 2)
    recon = torch.ones(1, 1, 2, 2)                # error 1 everywhere
    valid = torch.ones(1, 1, 2, 2)
    assert abs(masked_recon_loss(recon, x, valid).item() - 1.0) < 1e-4   # mean SE = 1


def test_kl_standard_normal_is_zero():
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)                    # var = 1 -> N(0,1) -> KL 0
    assert abs(kl_loss(mu, logvar).item()) < 1e-6


def test_kl_positive_off_standard():
    assert kl_loss(torch.ones(4, 16), torch.zeros(4, 16)).item() > 0    # shifted mean -> KL > 0
