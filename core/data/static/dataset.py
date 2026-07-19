"""GPU-resident tensors for the reconstruction models.

The whole processed split fits in VRAM (train-good ≈ 2 GB at 256²), so we load it
onto the GPU once and train whole-epoch on-device — no DataLoader, no per-batch
CPU↔GPU copy. Shuffle indices + slice on the GPU. (The acoustic-engine trick.)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import torch
from jaxtyping import Float

from core.data.static import preprocess as pp
from core.data.static import store


class GpuSplit:  # pragma: no cover  reads the processed store from disk; _stack_channels is the pure core
    """All samples of a query as GPU-resident tensors:
    x [N,C,H,W] (the channels), valid [N,1,H,W], gt [N,1,H,W]."""

    def __init__(
        self,
        df: pl.DataFrame,
        channels: tuple[str, ...] = ("xyz",),
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.df = df
        self.channels = tuple(channels)
        xs: list[np.ndarray] = []
        ms: list[np.ndarray] = []
        gs: list[np.ndarray] = []
        for path in df["path"].to_list():
            a = store.Store.load_arrays(path)
            xs.append(GpuSplit._stack_channels(a, channels))
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

    @staticmethod
    def _stack_channels(
        a: dict[str, np.ndarray],
        channels: Sequence[str],
    ) -> Float[np.ndarray, "c h w"]:
        """(H,W,3) channels -> one (C,H,W) float32 array."""
        return np.concatenate([a[c].transpose(2, 0, 1) for c in channels], 0).astype(
            np.float32
        )

    @staticmethod
    def load_split(  # noqa: PLR0913  # pragma: no cover
        split: str | None = None,
        label: str | None = None,
        cats: list[str] | None = None,
        channels: tuple[str, ...] = ("xyz",),
        device: str = "cuda",
        size: int | None = None,
        source: store.Source | None = None,
    ) -> "GpuSplit":
        """Query the store (filter by split / label / category) -> GPU-resident tensors."""
        df = store.Store.load(size=size or pp.SIZE, source=source)
        if cats:
            df = df.filter(pl.col("category").is_in(cats))
        if split is not None:
            df = df.filter(pl.col("split") == split)
        if label is not None:
            df = df.filter(pl.col("label") == label)
        if len(df) == 0:
            raise ValueError(f"empty split (split={split}, label={label}, cats={cats}) — build the store first")
        return GpuSplit(df, channels=channels, device=device)
