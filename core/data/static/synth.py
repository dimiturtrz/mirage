"""Synthetic-defect reader (the second adapter) — enumerate + load Isaac/Replicator renders.

The engine in `sim/` (its own env) writes each rendered sample in the SAME on-disk layout the MVTec
adapter reads, so the unified store ingests real + synthetic identically — the 2-source cloud whose
contrast IS the sim-to-real gap. Per category, under `<data root>/synth/`:

    <cat>/{train,validation}/good/{rgb/NNN.png, xyz/NNN.tiff}
    <cat>/test/<defect>/{rgb/NNN.png, xyz/NNN.tiff, gt/NNN.png}

`xyz` is a per-pixel position map (meters), the same representation the Zivid scanner produces for
MVTec — Replicator emits it from the depth/world-position pass so geometry channels match. By keeping
the format identical, `core/data/store.py` + preprocess + the harness work on synthetic unchanged;
only the *source root* differs. (Sim-only metadata — material / lighting / defect params — can ride
alongside as a sidecar later; the core triple is what the perception pipeline consumes.)
"""
from __future__ import annotations

from pathlib import Path

from core import config
from core.data.static import mvtec
from core.data.static.mvtec import Mvtec, Sample, Split  # identical on-disk format -> reuse the loader + Sample type

SPLITS = mvtec.SPLITS


class Synth:
    """Synthetic-defect reader (the second adapter) — delegates to the MVTec adapter over the synth root.

    Same shape as Mvtec: the synth `root` is the adapter's identity (default = synth_root) held in
    __init__; categories/samples build an Mvtec bound to that root and delegate."""

    def __init__(self, root: Path | None = None):
        self.root = root or self.synth_root()

    @staticmethod
    def synth_root() -> Path:
        """`<data root>/synth` — the engine's output root (sibling to raw/ + processed/)."""
        return Path(config.Config.data_root()) / "synth"

    def categories(self) -> list[str]:
        """Rendered categories under the synth root ([] until the engine has written any)."""
        return Mvtec(self.root).categories()

    def samples(self, cats: list[str] | None = None) -> list[Sample]:
        """Every rendered (category, split, defect, idx) sample — same walk + on-disk format as
        Mvtec.samples (only the source root differs), so the store ingests real + synth identically."""
        return Mvtec(self.root).samples(cats)


load_raw = Mvtec.load_raw  # identical on-disk format -> reuse the loader

__all__ = ["Sample", "Split", "Synth", "load_raw"]
