"""Device policy — decided once at the root, injected down the tree.

The root (a script's main / an experiment entry) calls `pick_device()` once and passes the string
down; nothing deeper probes `torch.cuda.is_available()` or hardcodes "cuda". `autocast` is the same
context on cuda (bf16) and cpu (a no-op), so one code path runs GPU-resident or edge/CPU unchanged.
"""
from __future__ import annotations

import torch


class Compute:
    """Device policy — decided once at the root (helpers folded in as staticmethods, public names kept)."""

    @staticmethod
    def pick_device(prefer: str = "cuda") -> str:
        """Root-level device choice: `prefer` if it's available, else cpu. Call once, inject the result."""
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        return prefer if prefer == "cpu" else "cpu"

    @staticmethod
    def autocast(x_or_dev, *, amp: bool = True, dtype: torch.dtype = torch.bfloat16):
        """bf16 autocast on cuda; a no-op elsewhere. Accepts a tensor or a device string."""
        dev = x_or_dev.device.type if isinstance(x_or_dev, torch.Tensor) else str(x_or_dev)
        return torch.autocast(dev, dtype=dtype, enabled=amp and dev == "cuda")

    @staticmethod
    def enable_tf32() -> None:
        """Use TF32 for fp32 matmuls (Ampere+): faster with negligible accuracy loss. Set once per run."""
        torch.set_float32_matmul_precision("high")
