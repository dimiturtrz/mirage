"""Conv VAE on the position-map grid — model #2, the deployable reconstruction baseline.

Symmetric strided-conv encoder/decoder; reconstructs the input channels (xyz, optionally
+rgb). Anomaly scoring (residual map + latent Mahalanobis) lives in evaluation/, not here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _enc_block(ci, co, p=0.0):
    layers = [nn.Conv2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.SiLU()]
    if p > 0:
        layers.append(nn.Dropout2d(p))
    return nn.Sequential(*layers)


def _dec_block(ci, co, last=False, p=0.0):
    layers = [nn.ConvTranspose2d(ci, co, 4, 2, 1)]
    if not last:
        layers += [nn.BatchNorm2d(co), nn.SiLU()]
        if p > 0:
            layers.append(nn.Dropout2d(p))
    return nn.Sequential(*layers)


class ConvVAE(nn.Module):
    def __init__(self, in_ch=3, base=32, latent=256, size=256, depth=5, dropout=0.0):
        super().__init__()
        self.in_ch, self.size, self.depth = in_ch, size, depth
        chs = [in_ch] + [base * 2 ** i for i in range(depth)]   # e.g. 3,32,64,128,256,512
        self.enc = nn.ModuleList(_enc_block(chs[i], chs[i + 1], dropout) for i in range(depth))
        self.feat = size // (2 ** depth)                        # 256 / 32 = 8
        self.bottleneck_ch = chs[-1]
        flat = chs[-1] * self.feat * self.feat
        self.fc_mu = nn.Linear(flat, latent)
        self.fc_logvar = nn.Linear(flat, latent)
        self.fc_dec = nn.Linear(latent, flat)
        self.z_drop = nn.Dropout(dropout)                       # drop the latent -> can't memorize/copy
        dchs = list(reversed(chs))                              # 512,256,...,in_ch
        self.dec = nn.ModuleList(
            _dec_block(dchs[i], dchs[i + 1], last=(i == depth - 1), p=dropout) for i in range(depth))

    def encode(self, x):
        for b in self.enc:
            x = b(x)
        x = x.flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        x = self.fc_dec(self.z_drop(z)).view(-1, self.bottleneck_ch, self.feat, self.feat)
        for b in self.dec:
            x = b(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
