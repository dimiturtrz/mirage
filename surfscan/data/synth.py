"""Synthetic-defect reader (the second adapter) — enumerate + load Isaac/Replicator renders.

The engine in `sim/` (its own env) writes each rendered sample in the SAME on-disk layout the MVTec
adapter reads, so the unified store ingests real + synthetic identically — the 2-source cloud whose
contrast IS the sim-to-real gap. Per category, under `<data root>/synth/`:

    <cat>/{train,validation}/good/{rgb/NNN.png, xyz/NNN.tiff}
    <cat>/test/<defect>/{rgb/NNN.png, xyz/NNN.tiff, gt/NNN.png}

`xyz` is a per-pixel position map (meters), the same representation the Zivid scanner produces for
MVTec — Replicator emits it from the depth/world-position pass so geometry channels match. By keeping
the format identical, `surfscan/data/store.py` + preprocess + the harness work on synthetic unchanged;
only the *source root* differs. (Sim-only metadata — material / lighting / defect params — can ride
alongside as a sidecar later; the core triple is what the perception pipeline consumes.)
"""
from __future__ import annotations

from pathlib import Path

from surfscan import config
from surfscan.data.mvtec import Sample, load_raw  # identical on-disk format -> reuse the loader

SPLITS = ("train", "validation", "test")


def synth_root() -> Path:
    """`<data root>/synth` — the engine's output root (sibling to raw/ + processed/)."""
    return Path(config.data_root()) / "synth"


def categories(root: Path | None = None) -> list[str]:
    root = Path(root or synth_root())
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def samples(root: Path | None = None, cats: list[str] | None = None) -> list[Sample]:
    """Enumerate every rendered (category, split, defect, idx) sample — same shape as mvtec.samples."""
    root = Path(root or synth_root())
    cats = cats or categories(root)
    out: list[Sample] = []
    for cat in cats:
        for split in SPLITS:
            sdir = root / cat / split
            if not sdir.is_dir():
                continue
            for ddir in sorted(p for p in sdir.iterdir() if p.is_dir()):
                gt_dir = ddir / "gt"
                for rgb in sorted((ddir / "rgb").glob("*.png")):
                    gt = gt_dir / f"{rgb.stem}.png"
                    out.append(Sample(
                        cat, split, ddir.name, int(rgb.stem),
                        rgb, ddir / "xyz" / f"{rgb.stem}.tiff",
                        gt if gt.exists() else None,
                    ))
    return out


__all__ = ["Sample", "load_raw", "samples", "categories", "synth_root"]
