"""Reconstruction-VAE losses — masked recon (valid pixels only) + KL."""
from __future__ import annotations

import torch
from jaxtyping import Bool, Float
from torch import Tensor


class Losses:
    """Reconstruction-VAE losses — masked recon (valid pixels only) + KL."""

    @staticmethod
    def masked_recon_loss(recon: Float[Tensor, "n c h w"],
                          x: Float[Tensor, "n c h w"],
                          valid: Bool[Tensor, "n 1 h w"]) -> Tensor:
        """Mean squared error over valid (object) pixels only — don't spend capacity on background.
        valid is [N,1,H,W] and broadcasts over the channels."""
        se = (recon - x) ** 2 * valid
        return se.sum() / (valid.sum() * x.shape[1] + 1e-8)

    @staticmethod
    def kl_loss(mu: Float[Tensor, "n ..."],
                logvar: Float[Tensor, "n ..."]) -> Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
