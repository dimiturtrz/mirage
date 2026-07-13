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
from core.data import mvtec
from core.data.mvtec import Sample, load_raw  # identical on-disk format -> reuse the loader + Sample type

SPLITS = mvtec.SPLITS


def synth_root() -> Path:
    """`<data root>/synth` — the engine's output root (sibling to raw/ + processed/)."""
    return Path(config.data_root()) / "synth"


def categories(root: Path | None = None) -> list[str]:
    """Rendered categories under the synth root ([] until the engine has written any)."""
    return mvtec.categories(root or synth_root())


def samples(root: Path | None = None, cats: list[str] | None = None) -> list[Sample]:
    """Every rendered (category, split, defect, idx) sample — same walk + on-disk format as
    mvtec.samples (only the source root differs), so the store ingests real + synth identically."""
    return mvtec.samples(root or synth_root(), cats)


__all__ = ["Sample", "load_raw", "samples", "categories", "synth_root"]
