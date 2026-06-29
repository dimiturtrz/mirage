"""Raw MVTec 3D-AD reader (the adapter) — enumerate samples + load raw channels.

On disk (per category):
    <cat>/{train,validation}/good/{rgb/NNN.png, xyz/NNN.tiff}
    <cat>/test/<defect>/{rgb/NNN.png, xyz/NNN.tiff, gt/NNN.png}
`gt` exists only under test/. defect == "good" -> normal; anything else -> anomaly.

This stays a raw->arrays reader; surfscan/data/store.py turns it into the unified store.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from surfscan import config

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Sample:
    category: str
    split: str            # train | validation | test
    defect: str           # good | hole | crack | contamination | combined | ...
    idx: int
    rgb_path: Path
    xyz_path: Path
    gt_path: Path | None   # None for train/validation (no gt shipped)

    @property
    def sample_id(self) -> str:
        return f"{self.category}_{self.split}_{self.defect}_{self.idx:03d}"

    @property
    def label(self) -> int:
        return 0 if self.defect == "good" else 1


def categories(root: Path | None = None) -> list[str]:
    root = Path(root or config.mvtec_root())
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def samples(root: Path | None = None, cats: list[str] | None = None) -> list[Sample]:
    """Enumerate every (category, split, defect, idx) sample on disk."""
    root = Path(root or config.mvtec_root())
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


def load_raw(s: Sample):
    """-> (rgb [H,W,3] uint8, xyz [H,W,3] float32, gt [H,W] uint8 | None)."""
    rgb = np.asarray(Image.open(s.rgb_path))
    xyz = tifffile.imread(s.xyz_path).astype(np.float32)
    gt = np.asarray(Image.open(s.gt_path)) if s.gt_path is not None else None
    return rgb, xyz, gt
