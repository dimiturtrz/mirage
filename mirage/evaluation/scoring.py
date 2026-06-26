"""Turn a reconstruction model into anomaly scores.

Per-pixel reconstruction residual -> a pixel anomaly map; the top-k residuals per image ->
an image-level score. (Latent Mahalanobis is the second score, bead synthscape-6v3 — added next.)
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def anomaly_maps(model, data, batch=64, amp=True):
    """-> (N,H,W) per-pixel squared-reconstruction-error maps, zeroed outside the object."""
    model.eval()
    out = []
    for i in range(0, len(data), batch):
        x = data.x[i:i + batch].to(memory_format=torch.channels_last)
        m = data.valid[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            recon, _, _ = model(x)
        err = ((recon.float() - x.float()) ** 2).sum(1, keepdim=True) * m   # N,1,H,W
        out.append(err.squeeze(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def inpaint_maps(model, data, grid=8, batch=16, amp=True):
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
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    r = model(xm.to(memory_format=torch.channels_last))
                inpainted[..., ys, xs] = r.float()[..., ys, xs]
        err = ((x - inpainted) ** 2).sum(1, keepdim=True) * m
        out.append(err.squeeze(1).cpu())
    return torch.cat(out).numpy()


def image_scores(amaps, valids, k=128):
    """Image-level score = mean of the top-k pixel residuals over valid pixels."""
    s = np.zeros(len(amaps), dtype=np.float64)
    for i, (a, v) in enumerate(zip(amaps, valids)):
        vals = a[v.astype(bool)]
        if vals.size == 0:
            continue
        s[i] = np.sort(vals)[-min(k, vals.size):].mean()
    return s
