"""Reconstruction-VAE losses — masked recon (valid pixels only) + KL."""
from __future__ import annotations

import torch


def masked_recon_loss(recon, x, valid):
    """Mean squared error over valid (object) pixels only — don't spend capacity on background.
    valid is [N,1,H,W] and broadcasts over the channels."""
    se = (recon - x) ** 2 * valid
    return se.sum() / (valid.sum() * x.shape[1] + 1e-8)


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
