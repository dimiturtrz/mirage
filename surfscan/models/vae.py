"""Conv VAE on the position-map grid — model #2, the deployable reconstruction baseline.

Symmetric strided-conv encoder/decoder; reconstructs the input channels (xyz, optionally
+rgb). Anomaly scoring (residual map + latent Mahalanobis) lives in evaluation/, not here.
"""
from __future__ import annotations

from typing import override

import torch
import torch.nn as nn
from jaxtyping import Float

from surfscan.training.hparams import ModelCfg


class ConvVAE(nn.Module):
    @staticmethod
    def enc_block(ci: int, co: int, p: float = 0.0) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Conv2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.SiLU()]
        if p > 0:
            layers.append(nn.Dropout2d(p))
        return nn.Sequential(*layers)

    @staticmethod
    def dec_block(ci: int, co: int, *, last: bool = False, p: float = 0.0) -> nn.Sequential:
        layers: list[nn.Module] = [nn.ConvTranspose2d(ci, co, 4, 2, 1)]
        if not last:
            layers += [nn.BatchNorm2d(co), nn.SiLU()]
            if p > 0:
                layers.append(nn.Dropout2d(p))
        return nn.Sequential(*layers)

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        in_ch, base, latent, size, depth, dropout = (
            cfg.in_ch, cfg.base, cfg.latent, cfg.size, cfg.depth, cfg.dropout)
        self.in_ch, self.size, self.depth = in_ch, size, depth
        channels = [in_ch] + [base * 2 ** level for level in range(depth)]   # e.g. 3,32,64,128,256,512
        self.enc = nn.ModuleList(
            ConvVAE.enc_block(channels[level], channels[level + 1], dropout) for level in range(depth))
        self.feat = size // (2 ** depth)                        # 256 / 32 = 8
        self.bottleneck_ch = channels[-1]
        flat_dim = channels[-1] * self.feat * self.feat
        self.fc_mu = nn.Linear(flat_dim, latent)
        self.fc_logvar = nn.Linear(flat_dim, latent)
        self.fc_dec = nn.Linear(latent, flat_dim)
        self.z_drop = nn.Dropout(dropout)                       # drop the latent -> can't memorize/copy
        decoder_channels = list(reversed(channels))            # 512,256,...,in_ch
        self.dec = nn.ModuleList(
            ConvVAE.dec_block(decoder_channels[level], decoder_channels[level + 1],
                              last=(level == depth - 1), p=dropout) for level in range(depth))

    def encode(
        self, x: Float[torch.Tensor, "b c h w"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for b in self.enc:
            x = b(x)
        x = x.flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(
        self, mu: Float[torch.Tensor, "b latent"], logvar: Float[torch.Tensor, "b latent"]
    ) -> Float[torch.Tensor, "b latent"]:
        if not self.training:
            return mu
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z: Float[torch.Tensor, "b latent"]) -> Float[torch.Tensor, "b c h w"]:
        x = self.fc_dec(self.z_drop(z)).view(-1, self.bottleneck_ch, self.feat, self.feat)
        for b in self.dec:
            x = b(x)
        return x

    @override
    def forward(
        self, x: Float[torch.Tensor, "b c h w"]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
