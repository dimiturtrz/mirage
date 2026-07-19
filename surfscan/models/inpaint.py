"""Masked-inpainting autoencoder (RIAD-style) — reconstruction that can't copy.

Train: mask random patches of the input, reconstruct the full image from the surrounding
context. Score: systematically mask each grid cell, fill it from the model's prediction, assemble
an 'inpainted' image; anomaly = (input - inpainted)^2. Because a hidden region can't be copied,
an anomaly there gets predicted-as-normal -> the residual spikes only at the anomaly (localizes).
"""
from __future__ import annotations

from typing import override

import torch
import torch.nn as nn
from jaxtyping import Float

from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import ModelCfg


class InpaintAE(nn.Module):
    @staticmethod
    def random_mask(
        n: int, size: int, patch: int, ratio: float, device: str | torch.device
    ) -> Float[torch.Tensor, "n 1 size size"]:
        """(n,1,size,size) mask: ~ratio of patch-cells zeroed (0 = masked/hidden, 1 = visible)."""
        g = size // patch
        keep = (torch.rand(n, 1, g, g, device=device) > ratio).float()
        return keep.repeat_interleave(patch, 2).repeat_interleave(patch, 3)

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        in_ch, base, latent, size, depth, dropout = (
            cfg.in_ch, cfg.base, cfg.latent, cfg.size, cfg.depth, cfg.dropout)
        self.in_ch, self.size, self.depth = in_ch, size, depth
        channels = [in_ch] + [base * 2 ** level for level in range(depth)]
        self.enc = nn.ModuleList(
            ConvVAE.enc_block(channels[level], channels[level + 1], dropout) for level in range(depth))
        self.feat = size // (2 ** depth)
        self.bottleneck_ch = channels[-1]
        flat_dim = channels[-1] * self.feat * self.feat
        self.fc = nn.Sequential(nn.Linear(flat_dim, latent), nn.SiLU(), nn.Linear(latent, flat_dim))
        decoder_channels = list(reversed(channels))
        self.dec = nn.ModuleList(
            ConvVAE.dec_block(decoder_channels[level], decoder_channels[level + 1],
                              last=(level == depth - 1), p=dropout) for level in range(depth))

    @override
    def forward(self, x: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h w"]:
        for b in self.enc:
            x = b(x)
        x = self.fc(x.flatten(1)).view(-1, self.bottleneck_ch, self.feat, self.feat)
        for b in self.dec:
            x = b(x)
        return x
