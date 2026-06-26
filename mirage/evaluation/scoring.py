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


def image_scores(amaps, valids, k=128):
    """Image-level score = mean of the top-k pixel residuals over valid pixels."""
    s = np.zeros(len(amaps), dtype=np.float64)
    for i, (a, v) in enumerate(zip(amaps, valids)):
        vals = a[v.astype(bool)]
        if vals.size == 0:
            continue
        s[i] = np.sort(vals)[-min(k, vals.size):].mean()
    return s
