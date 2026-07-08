"""The feature-space autoencoder — pure torch, no backbone dependency.

Split out from feat_recon.py so the AE (a small conv encoder/decoder on a C×h×w feature map) imports
and tests without torchvision; the frozen backbone that produces the feature map lives in FeatExtractor.
"""
from __future__ import annotations

import torch.nn as nn


class FeatAE(nn.Module):
    """Small conv AE on a C×h×w feature map."""
    def __init__(self, ch, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(ch, hidden, 3, 2, 1), nn.BatchNorm2d(hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1), nn.BatchNorm2d(hidden), nn.SiLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.BatchNorm2d(hidden), nn.SiLU(),
            nn.ConvTranspose2d(hidden, ch, 4, 2, 1))

    def forward(self, x):
        return self.dec(self.enc(x))
