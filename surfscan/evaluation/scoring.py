"""Turn a reconstruction model into anomaly scores.

Two image-level scores from a reconstruction model:
  - residual: per-pixel squared-recon-error -> a pixel anomaly map; top-k residuals -> an image score.
  - latent Mahalanobis: distance of the encoder latent to the Gaussian of NORMAL latents. A defect the
    decoder happens to rebuild well (low residual) can still sit off the normal-latent cloud — this
    catches it. Ledoit-Wolf shrinkage keeps the precision well-conditioned when the normal latents are
    fewer than the latent dim.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from core.compute import autocast


@torch.no_grad()
def anomaly_maps(model, data, batch=64, *, amp=True):
    """-> (N,H,W) per-pixel squared-reconstruction-error maps, zeroed outside the object."""
    model.eval()
    out = []
    for i in range(0, len(data), batch):
        x = data.x[i:i + batch].to(memory_format=torch.channels_last)
        m = data.valid[i:i + batch]
        with autocast(x, amp=amp):
            recon, _, _ = model(x)
        err = ((recon.float() - x.float()) ** 2).sum(1, keepdim=True) * m   # N,1,H,W
        out.append(err.squeeze(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def inpaint_maps(model, data, grid=8, batch=16, *, amp=True):
    """Anomaly maps for the inpainting model: mask each grid cell, fill from the model's
    prediction, assemble an 'inpainted' image; anomaly = (input - inpainted)^2."""
    model.eval()
    size = data.x.shape[-1]
    patch = size // grid
    out = []
    for i in range(0, len(data), batch):
        x = data.x[i:i + batch]
        m = data.valid[i:i + batch]
        inpainted = torch.zeros_like(x)
        for gy in range(grid):
            for gx in range(grid):
                ys = slice(gy * patch, (gy + 1) * patch)
                xs = slice(gx * patch, (gx + 1) * patch)
                xm = x.clone()
                xm[..., ys, xs] = 0
                with autocast(xm, amp=amp):
                    r = model(xm.to(memory_format=torch.channels_last))
                inpainted[..., ys, xs] = r.float()[..., ys, xs]
        err = ((x - inpainted) ** 2).sum(1, keepdim=True) * m
        out.append(err.squeeze(1).cpu())
    return torch.cat(out).numpy()


def image_scores(amaps, valids, k=128):
    """Image-level score = mean of the top-k pixel residuals over valid pixels."""
    s = np.zeros(len(amaps), dtype=np.float64)
    for i, (a, v) in enumerate(zip(amaps, valids, strict=True)):
        vals = a[v.astype(bool)]
        if vals.size == 0:
            continue
        s[i] = np.sort(vals)[-min(k, vals.size):].mean()
    return s


@torch.no_grad()
def latents(model, data, batch=64, *, amp=True):
    """-> (N,D) encoder latent means (VAE mu) — the input to latent-distance scoring."""
    model.eval()
    out = []
    for i in range(0, len(data), batch):
        x = data.x[i:i + batch].to(memory_format=torch.channels_last)
        with autocast(x, amp=amp):
            mu, _ = model.encode(x)
        out.append(mu.float().cpu())
    return torch.cat(out).numpy()


def mahalanobis(train_lat, test_lat):
    """Fit a Gaussian on the NORMAL (train) latents; score each test latent by its Mahalanobis
    distance to it. Ledoit-Wolf shrinkage -> a well-conditioned precision even when n_normal < D."""
    lw = LedoitWolf().fit(train_lat)
    return np.sqrt(lw.mahalanobis(test_lat))          # lw centers on the fitted (normal) mean
