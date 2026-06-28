"""BTF (FPFH geometry memory bank) over categories -> aggregate. The geometry SOTA-floor.

Uses the eval harness. Fit a normal-FPFH bank per category, score the test split. No training.

Run:  python -m mirage.training.run_btf [--cats bagel ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from mirage.data import store
from mirage.evaluation import harness, scoring
from mirage.models.fpfh_bank import FpfhBank


def _load(cat, split, size, label=None):
    df = store.load(size=size).filter(pl.col("category") == cat).filter(pl.col("split") == split)
    if label is not None:
        df = df.filter(pl.col("label") == label)
    return df, [store.load_arrays(p) for p in df["path"].to_list()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--size", type=int, default=None, help="store resolution (default 256; 512 = native-ish)")
    ap.add_argument("--out", type=Path, default=Path("runs/btf"))
    args = ap.parse_args()
    size = args.size or 256

    def fit(c):
        _, train = _load(c, "train", size, 0)
        return FpfhBank().fit([(a["xyz"], a["valid"]) for a in train])

    def score(bank, c):
        dft, test = _load(c, "test", size)
        amaps = np.stack([bank.score_map(a["xyz"], a["valid"]) for a in test])
        valids = np.stack([a["valid"].astype(bool) for a in test])
        masks = np.stack([(a["gt"] > 0) for a in test])
        scores = scoring.image_scores(amaps, valids)
        return (amaps, valids, masks, scores,
                dft["label"].to_numpy(), np.array(dft["defect"].to_list()))

    harness.run("btf_fpfh", fit, score, cats=args.cats, out=args.out)


if __name__ == "__main__":
    main()
