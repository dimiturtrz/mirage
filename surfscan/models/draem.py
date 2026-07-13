"""DRAEM — discriminatively-trained reconstruction with synthetic proxy anomalies (Stage-0).

Synthesize crude anomalies (a Perlin-ish blob filled with noise) on a normal sample. Train:
(1) a reconstructive U-Net to restore the normal, (2) a discriminative U-Net to segment the
synthetic anomaly from concat(augmented, reconstruction). Test: the discriminator output IS the
anomaly map. Self-contained — the synthesis is crude proxy noise, NOT the Stage-1 physics engine
(DAS3D/3DSR are the 3D SOTA heirs that swap in realistic synthesis).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize as sk_resize


class UNet(nn.Module):
    @staticmethod
    def _block(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, 1, 1), nn.BatchNorm2d(co), nn.SiLU(),
            nn.Conv2d(co, co, 3, 1, 1), nn.BatchNorm2d(co), nn.SiLU())

    def __init__(self, in_ch, out_ch, base=32):
        super().__init__()
        self.e1, self.e2 = UNet._block(in_ch, base), UNet._block(base, base * 2)
        self.e3, self.e4 = UNet._block(base * 2, base * 4), UNet._block(base * 4, base * 8)
        self.bott = UNet._block(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2); self.d4 = UNet._block(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2); self.d3 = UNet._block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2); self.d2 = UNet._block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2); self.d1 = UNet._block(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


class Draem(nn.Module):
    def __init__(self, ch, base=32):
        super().__init__()
        self.recon = UNet(ch, ch, base)
        self.disc = UNet(2 * ch, 1, base)

    def forward(self, x):
        rec = self.recon(x)
        logits = self.disc(torch.cat([x, rec], 1))
        return rec, logits


class DraemSynth:
    """The crude Perlin proxy-anomaly synthesis (public `synthesize` name kept as a staticmethod)."""

    @staticmethod
    def _perlin_mask(h, w, rng, res=16):
        low = rng.rand(res, res).astype(np.float32)
        m = sk_resize(low, (h, w), order=1, preserve_range=True)
        return (m > rng.uniform(0.6, 0.8)).astype(np.float32)

    @staticmethod
    def synthesize(x, valid, rng):
        """x: (B,C,H,W), valid: (B,1,H,W) -> (aug, mask) with a noise-filled blob inside the object."""
        B, C, H, W = x.shape
        masks = np.stack([DraemSynth._perlin_mask(H, W, rng) for _ in range(B)])[:, None]      # B,1,H,W
        m = torch.from_numpy(masks).to(x.device) * valid
        beta = float(rng.uniform(0.2, 1.0))
        noise = torch.randn_like(x) * x.std()
        aug = x * (1 - m) + (beta * noise + (1 - beta) * x) * m
        return aug, m
