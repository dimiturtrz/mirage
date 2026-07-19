"""Turn a reconstruction model into anomaly scores.

Two image-level scores from a reconstruction model:
  - residual: per-pixel squared-recon-error -> a pixel anomaly map; top-k residuals -> an image score.
  - latent Mahalanobis: distance of the encoder latent to the Gaussian of NORMAL latents. A defect the
    decoder happens to rebuild well (low residual) can still sit off the normal-latent cloud — this
    catches it. Ledoit-Wolf shrinkage keeps the precision well-conditioned when the normal latents are
    fewer than the latent dim.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from sklearn.covariance import LedoitWolf
from torch import Tensor, nn

from core.compute import Compute
from core.method import ScoreArrays

if TYPE_CHECKING:
    import polars as pl

    from core.data.static.dataset import GpuSplit


class Scoring:
    """Turn a reconstruction model into anomaly scores. The model-driven maps/latents share the fitted
    `model` + `amp` policy (state_candidates 6sc/5ak: carried by every model method) -> held in __init__;
    the array/stat reducers (image_scores/score_arrays/mahalanobis) touch no model, so they stay static.
    `batch` is a per-call arg, not state — its default legitimately differs per method (64 vs inpaint's 16)."""

    def __init__(self, model: nn.Module, *, amp: bool = True):
        self.model = model
        self.amp = amp

    @torch.no_grad()
    def anomaly_maps(self, data: GpuSplit, batch: int = 64) -> Float[np.ndarray, "n h w"]:
        """-> (N,H,W) per-pixel squared-reconstruction-error maps, zeroed outside the object."""
        self.model.eval()

        def span(i0: int, i1: int):
            x = data.x[i0:i1].to(memory_format=torch.channels_last)
            m = data.valid[i0:i1]
            with Compute.autocast(x, amp=self.amp):
                recon, _, _ = self.model(x)
            err = ((recon.float() - x.float()) ** 2).sum(1, keepdim=True) * m   # N,1,H,W
            return err.squeeze(1).cpu()
        return Compute.batched_forward(span, len(data), batch).numpy()

    @torch.no_grad()
    def inpaint_maps(self, data: GpuSplit, grid: int = 8, batch: int = 16) -> Float[np.ndarray, "n h w"]:
        """Anomaly maps for the inpainting model: mask each grid cell, fill from the model's
        prediction, assemble an 'inpainted' image; anomaly = (input - inpainted)^2."""
        self.model.eval()
        size = data.x.shape[-1]
        patch = size // grid

        def span(i0: int, i1: int):
            x = data.x[i0:i1]
            m = data.valid[i0:i1]
            inpainted = torch.zeros_like(x)
            for gy in range(grid):
                for gx in range(grid):
                    ys = slice(gy * patch, (gy + 1) * patch)
                    xs = slice(gx * patch, (gx + 1) * patch)
                    xm = x.clone()
                    xm[..., ys, xs] = 0
                    with Compute.autocast(xm, amp=self.amp):
                        r = self.model(xm.to(memory_format=torch.channels_last))
                    inpainted[..., ys, xs] = r.float()[..., ys, xs]
            err = ((x - inpainted) ** 2).sum(1, keepdim=True) * m
            return err.squeeze(1).cpu()
        return Compute.batched_forward(span, len(data), batch).numpy()

    @staticmethod
    def image_scores(amaps: Float[np.ndarray, "n h w"], valids: Bool[np.ndarray, "n h w"],
                     k: int = 128) -> Float[np.ndarray, "n"]:
        """Image-level score = mean of the top-k pixel residuals over valid pixels."""
        s = np.zeros(len(amaps), dtype=np.float64)
        for i, (a, v) in enumerate(zip(amaps, valids, strict=True)):
            vals = a[v.astype(bool)]
            if vals.size == 0:
                continue
            s[i] = np.sort(vals)[-min(k, vals.size):].mean()
        return s

    @staticmethod
    def score_arrays(amaps: Float[np.ndarray, "n h w"], test: GpuSplit,
                     scores: Float[np.ndarray, "n"] | None = None,
                     idx: Int[np.ndarray, "k"] | None = None) -> ScoreArrays:
        """GpuSplit path -> the harness ScoreArrays 6-tuple. `idx` selects a row subset (the triad arm
        scores only the shared real-eval half); default = the whole split. Default image score = top-k
        residual over valid px (pass `scores` to override)."""
        sel = slice(None) if idx is None else torch.as_tensor(idx, device=test.valid.device)
        valids = test.valid[sel].squeeze(1).cpu().numpy().astype(bool)
        masks = test.gt[sel].squeeze(1).cpu().numpy().astype(bool)
        labels, defects = test.df["label"].to_numpy(), np.array(test.df["defect"].to_list())
        if idx is not None:
            labels, defects = labels[idx], defects[idx]
        scores = Scoring.image_scores(amaps, valids) if scores is None else scores
        return ScoreArrays(amaps, valids, masks, scores, labels, defects)

    @staticmethod
    def score_arrays_store(amaps: Float[np.ndarray, "n h w"], test: list[dict[str, Any]], dft: pl.DataFrame,
                           scores: Float[np.ndarray, "n"] | None = None) -> ScoreArrays:
        """Store-dict path -> the same ScoreArrays 6-tuple: valid/gt come from per-sample store dicts (the
        BTF / PointMAE geometry methods read the raw store, not a GpuSplit); labels from the dataframe."""
        valids = np.stack([a["valid"].astype(bool) for a in test])
        masks = np.stack([a["gt"] > 0 for a in test])
        scores = Scoring.image_scores(amaps, valids) if scores is None else scores
        return ScoreArrays(amaps, valids, masks, scores,
                           dft["label"].to_numpy(), np.array(dft["defect"].to_list()))

    @staticmethod
    def logits_to_amap(logits: Float[Tensor, "n 1 h w"],
                       valid: Float[Tensor, "n 1 h w"]) -> Float[Tensor, "n h w"]:
        """Segmentation logits -> a per-pixel anomaly map: sigmoid, gated by the valid mask, object-frame,
        (N,1,H,W) -> (N,H,W) on cpu. The shared amap tail of the discriminative (DRAEM / triad) scorers;
        numpy-wanting callers append `.numpy()`."""
        return (torch.sigmoid(logits.float()) * valid).squeeze(1).cpu()

    @staticmethod
    @torch.no_grad()
    def score_logits(forward: Callable[[int, int], Float[Tensor, "b 1 h w"]],
                     valid: Float[Tensor, "n 1 h w"], test: GpuSplit, batch: int,
                     idx: Int[np.ndarray, "k"] | None = None) -> ScoreArrays:
        """Discriminative scorer (DRAEM / triad): batched forward -> sigmoid amap tail -> ScoreArrays.
        `forward(i0, i1)` returns the model's raw segmentation logits for that row-span; every model-specific
        part (tuple unpack, autocast, channels_last, the triad's eval-half subset) lives in that closure.
        `valid` is the per-pixel mask aligned to forward's row space (its length is the row count); `idx`
        subsets the split rows for score_arrays (the triad scores only the shared real-eval half; DRAEM
        scores the whole split)."""
        def span(i0: int, i1: int):
            return Scoring.logits_to_amap(forward(i0, i1), valid[i0:i1])
        amaps = Compute.batched_forward(span, valid.shape[0], batch).numpy()
        return Scoring.score_arrays(amaps, test, idx=idx)

    @torch.no_grad()
    def latents(self, data: GpuSplit, batch: int = 64) -> Float[np.ndarray, "n d"]:
        """-> (N,D) encoder latent means (VAE mu) — the input to latent-distance scoring."""
        self.model.eval()

        def span(i0: int, i1: int):
            x = data.x[i0:i1].to(memory_format=torch.channels_last)
            with Compute.autocast(x, amp=self.amp):
                # pyrefly: ignore[not-callable]  torch's nn.Module.__getattr__ stub returns Tensor|Module,
                # erasing the concrete subclass method; ConvVAE.encode resolves normally at runtime
                mu, _ = self.model.encode(x)
            return mu.float().cpu()
        return Compute.batched_forward(span, len(data), batch).numpy()

    @staticmethod
    def mahalanobis(train_lat: Float[np.ndarray, "n d"], test_lat: Float[np.ndarray, "m d"]) -> Float[np.ndarray, "m"]:
        """Fit a Gaussian on the NORMAL (train) latents; score each test latent by its Mahalanobis
        distance to it. Ledoit-Wolf shrinkage -> a well-conditioned precision even when n_normal < D."""
        lw = LedoitWolf().fit(train_lat)
        return np.sqrt(lw.mahalanobis(test_lat))          # lw centers on the fitted (normal) mean
