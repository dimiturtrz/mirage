"""GPU-resident tensors for the reconstruction models.

The whole processed split fits in VRAM (train-good ≈ 2 GB at 256²), so we load it
onto the GPU once and train whole-epoch on-device — no DataLoader, no per-batch
CPU↔GPU copy. Shuffle indices + slice on the GPU. (The acoustic-engine trick.)
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch

from surfscan.data import preprocess as pp
from surfscan.data import store


def _stack_channels(a: dict, channels) -> np.ndarray:
    """(H,W,3) channels -> one (C,H,W) float32 array."""
    return np.concatenate([a[c].transpose(2, 0, 1) for c in channels], 0).astype(np.float32)


class GpuSplit:
    """All samples of a query as GPU-resident tensors:
    x [N,C,H,W] (the channels), valid [N,1,H,W], gt [N,1,H,W]."""

    def __init__(self, df: pl.DataFrame, channels=("xyz",), device="cuda", dtype=torch.float32):
        self.df = df
        self.channels = tuple(channels)
        xs, ms, gs = [], [], []
        for path in df["path"].to_list():
            a = store.load_arrays(path)
            xs.append(_stack_channels(a, channels))
            ms.append(a["valid"][None].astype(np.float32))
            gs.append((a["gt"] > 0)[None].astype(np.float32))
        self.x = torch.from_numpy(np.stack(xs)).to(device=device, dtype=dtype)
        self.valid = torch.from_numpy(np.stack(ms)).to(device=device, dtype=dtype)
        self.gt = torch.from_numpy(np.stack(gs)).to(device=device, dtype=dtype)

    @property
    def in_ch(self) -> int:
        return self.x.shape[1]

    def __len__(self) -> int:
        return self.x.shape[0]


def load_split(split=None, label=None, cats=None, channels=("xyz",),
               device="cuda", dtype=torch.float32, size=None) -> GpuSplit:
    """Query the store (filter by split / label / category) -> GPU-resident tensors."""
    df = store.load(size=size or pp.SIZE)
    if cats:
        df = df.filter(pl.col("category").is_in(cats))
    if split is not None:
        df = df.filter(pl.col("split") == split)
    if label is not None:
        df = df.filter(pl.col("label") == label)
    if len(df) == 0:
        raise ValueError(f"empty split (split={split}, label={label}, cats={cats}) — build the store first")
    return GpuSplit(df, channels=channels, device=device, dtype=dtype)
