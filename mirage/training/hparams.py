"""Validated hyperparameters (serialized to runs/<run>/config.json for reproducibility)."""
from __future__ import annotations

from pydantic import BaseModel


class HParams(BaseModel):
    cats: list[str] | None = None      # None = all categories
    channels: list[str] = ["xyz"]      # geometry-only; ["xyz","rgb"] = fused (6ch)
    size: int = 256
    base: int = 32
    latent: int = 256
    depth: int = 5
    dropout: float = 0.0               # >0 breaks the identity/copy shortcut (forces generalization)
    batch: int = 64
    epochs: int = 100
    lr: float = 2e-4
    beta: float = 1e-4                 # KL weight (small -> favor reconstruction quality)
    bf16: bool = True
    compile: bool = True
    seed: int = 0
