"""Validated hyperparameters (logged as the config.json MLflow artifact for reproducibility;
evaluate reads them back via tracking.load_config to rebuild the exact model)."""
from __future__ import annotations

from pydantic import BaseModel


class HParams(BaseModel):
    model_type: str = "vae"            # vae | inpaint
    cats: list[str] | None = None      # None = all categories
    channels: list[str] = ["xyz"]      # geometry-only; ["xyz","rgb"] = fused (6ch)
    mask_ratio: float = 0.5            # inpaint: fraction of patches masked during training
    grid: int = 8                     # inpaint: grid for masking (patch = size // grid)
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
