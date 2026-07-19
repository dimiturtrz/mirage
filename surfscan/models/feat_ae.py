"""The feature-space autoencoder — pure torch, no backbone dependency.

Split out from feat_recon.py so the AE (a small conv encoder/decoder on a C×h×w feature map) imports
and tests without torchvision; the frozen backbone that produces the feature map lives in FeatExtractor.
"""
from __future__ import annotations

from typing import override

import torch
import torch.nn as nn
from jaxtyping import Float


class FeatAE(nn.Module):
    """Small conv AE on a C×h×w feature map."""
    def __init__(self, ch: int, hidden: int = 128) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(ch, hidden, 3, 2, 1), nn.BatchNorm2d(hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1), nn.BatchNorm2d(hidden), nn.SiLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.BatchNorm2d(hidden), nn.SiLU(),
            nn.ConvTranspose2d(hidden, ch, 4, 2, 1))

    @override
    def forward(self, x: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h w"]:
        return self.dec(self.enc(x))
