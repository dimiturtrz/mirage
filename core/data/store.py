"""Consolidate raw MVTec 3D-AD into processed/mvtec3d/<paramkey>/ — the unified store.

    processed/mvtec3d/<paramkey>/
        data/<sample_id>.npz    # xyz [H,W,3], rgb [H,W,3], valid [H,W], gt [H,W]
        meta.csv                # one row per sample, the unified schema (polars)

`paramkey` encodes the preprocess params so two recipes never collide. The meta.csv
IS the inventory — filter it (category / split / label) to get any split; no separate
index. Mirrors systole's cardioseg/data/store.py.

CLI:  python -m core.data.store [--cats bagel ...] [--size 256] [--rebuild]
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from core import config
from core.data import mvtec
from core.data import preprocess as pp
from core.obs import Obs

log = Obs.get()

# unified cloud columns. `file` -> the npz in data/; `raw_*` -> the original channels.
META_FIELDS = ["sample_id", "category", "split", "defect", "label", "has_gt", "file",
               "raw_rgb", "raw_xyz", "raw_gt", "H", "W", "n_valid", "defect_frac"]


class Store:
    """Consolidate raw MVTec 3D-AD into processed/mvtec3d/<paramkey>/ — the unified store."""

    @staticmethod
    def param_key(size=pp.SIZE, nb=pp.SO_NB, std=pp.SO_STD) -> str:
        return f"s{size}_so{nb}-{str(std).replace('.', 'p')}"

    @staticmethod
    def dataset_dir(size=pp.SIZE, nb=pp.SO_NB, std=pp.SO_STD) -> Path:
        return config.Config.processed_dir() / "mvtec3d" / Store.param_key(size, nb, std)

    @staticmethod
    def _row(s: mvtec.Sample, arr: dict, file: str) -> dict:
        valid, gt = arr["valid"], arr["gt"]
        return {
            "sample_id": s.sample_id, "category": s.category, "split": s.split,
            "defect": s.defect, "label": s.label, "has_gt": s.gt_path is not None, "file": file,
            "raw_rgb": str(s.rgb_path), "raw_xyz": str(s.xyz_path),
            "raw_gt": str(s.gt_path) if s.gt_path is not None else "",
            "H": int(valid.shape[0]), "W": int(valid.shape[1]),
            "n_valid": int(valid.sum()), "defect_frac": float((gt > 0).mean()),
        }

    @staticmethod
    def _schema():
        def dt(k):
            if k in ("label", "H", "W", "n_valid"):
                return pl.Int64
            if k == "defect_frac":
                return pl.Float64
            if k == "has_gt":
                return pl.Boolean
            return pl.Utf8
        return {k: dt(k) for k in META_FIELDS}

    @staticmethod
    def build(p=None, cats=None, workers=None, *, rebuild=False) -> Path:  # pragma: no cover  disk write
        """Consolidate into processed/mvtec3d/<paramkey>/. Process-if-missing; (re)writes meta.csv."""
        p = p or pp.PP()
        out = Store.dataset_dir(p.size, p.nb, p.std)
        data_dir = out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        samples = mvtec.Mvtec.samples(cats=cats)
        todo = samples if rebuild else [s for s in samples if not (data_dir / f"{s.sample_id}.npz").exists()]

        def _one(s: mvtec.Sample):
            rgb, xyz, gt = mvtec.Mvtec.load_raw(s)
            arr = pp.Preprocess.preprocess(rgb, xyz, gt, p)
            np.savez_compressed(data_dir / f"{s.sample_id}.npz", **arr)

        if todo:
            workers = workers or max(1, (os.cpu_count() or 4) - 2)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_one, todo))

        rows = []
        for s in samples:
            f = f"{s.sample_id}.npz"
            if (data_dir / f).exists():
                rows.append(Store._row(s, dict(np.load(data_dir / f)), f))
        pl.DataFrame(rows, schema=Store._schema(), strict=False).write_csv(out / "meta.csv")
        return out

    @staticmethod
    def load(size=pp.SIZE, nb=pp.SO_NB, std=pp.SO_STD, cats=None, workers=None) -> pl.DataFrame:  # pragma: no cover
        """Ensure consolidated, return one polars frame (the cloud) with an absolute `path` column."""
        if size is None:                                    # callers may pass size=None -> use the default store
            size = pp.SIZE
        out = Store.dataset_dir(size, nb, std)
        if not (out / "meta.csv").exists():
            Store.build(pp.PP(size=size, nb=nb, std=std), cats=cats, workers=workers)   # build takes a PP
        df = pl.read_csv(out / "meta.csv", infer_schema_length=10000)
        return df.with_columns((pl.lit(str(out / "data")) + "/" + pl.col("file")).alias("path"))

    @staticmethod
    def load_arrays(path) -> dict:  # pragma: no cover  np.load from disk
        """Load one consolidated sample npz -> {xyz, rgb, valid, gt}."""
        z = np.load(path)
        return {k: z[k] for k in z.files}

    @staticmethod
    def arrays(cat, split, label=None, size=None):  # pragma: no cover  Store.load + load_arrays from disk
        """(df, [array-dict per sample]) for one category/split — the raw processed arrays
        (xyz/rgb/valid/gt). Used by the geometry methods that work on point clouds, not the GPU grid."""
        df = Store.load(size=size).filter(pl.col("category") == cat).filter(pl.col("split") == split)
        if label is not None:
            df = df.filter(pl.col("label") == label)
        return df, [Store.load_arrays(p) for p in df["path"].to_list()]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="consolidate MVTec 3D-AD -> processed/mvtec3d/<paramkey>/")
    ap.add_argument("--cats", nargs="*", default=None, help="categories (default: all)")
    ap.add_argument("--size", type=int, default=pp.SIZE)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    Store.build(pp.PP(size=args.size), cats=args.cats, rebuild=args.rebuild)  # process-if-missing + rewrite meta
    df = Store.load(size=args.size, cats=args.cats)
    log.info(f"=== mvtec3d cloud: {len(df)} samples ===")
    log.info(df.group_by("split", "label").agg(pl.len().alias("n")).sort("split", "label"))
    log.info(df.group_by("category").agg(pl.len().alias("n"), pl.col("label").sum().alias("anom")).sort("category"))
