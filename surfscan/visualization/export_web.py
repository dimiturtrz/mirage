"""Export samples to the web viewer's data/ — points + rgb + PatchCore anomaly + gt, downsampled.

Run: python -m surfscan.visualization.export_web --samples bagel:test:hole:0 cookie:test:crack:0
Writes pointcloud-viewer/data/<id>.json + manifest.json. The data is MVTec-derived (CC BY-NC-SA),
so pointcloud-viewer/data/ is gitignored — run this to populate the viewer locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.obs import get
from surfscan.visualization.show import load_processed, patchcore_map

log = get()

OUT = Path(__file__).resolve().parents[2] / "pointcloud-viewer" / "data"


def export_one(cat, split, defect, idx, n=15000, seed=0):
    rgb, xyz, gt, valid = load_processed(cat, split, defect, idx)
    v = valid.astype(bool)
    an = patchcore_map(cat, rgb, valid)[v]
    pts, cols, g = xyz[v], rgb[v], (gt[v] > 0).astype(int)
    an = (an / (np.percentile(an, 99) + 1e-9)).clip(0, 1)        # robust 0..1
    pts = pts - pts.mean(0)                                       # center for the viewer
    if len(pts) > n:
        s = np.random.RandomState(seed).choice(len(pts), n, replace=False)
        pts, cols, an, g = pts[s], cols[s], an[s], g[s]
    sid = f"{cat}_{split}_{defect}_{idx:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{sid}.json").write_text(json.dumps({
        "id": sid,
        "points": pts.round(4).flatten().tolist(),
        "rgb": cols.round(3).flatten().tolist(),
        "anomaly": an.round(3).tolist(),
        "gt": g.tolist(),
    }))
    log.info(f"  wrote {sid}  ({len(pts)} pts)")
    return sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True, help="cat:split:defect:idx ...")
    args = ap.parse_args()
    ids = []
    for spec in args.samples:
        cat, split, defect, idx = spec.split(":")
        ids.append(export_one(cat, split, defect, int(idx)))
    (OUT / "manifest.json").write_text(json.dumps({"samples": ids}, indent=2))
    log.info(f"manifest: {len(ids)} samples -> {OUT}")


if __name__ == "__main__":
    main()
