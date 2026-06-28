"""Fused PatchCore(rgb) + BTF(FPFH geometry) anomaly maps -> aggregate (M3DM-lite). Uses the harness.

Score each test sample with both banks, z-score each map family over the valid test pixels, sum.

Run:  python -m mirage.experiments.run_fused [--cats bagel ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mirage.data import store
from mirage.data.dataset import load_split
from mirage.evaluation import harness, scoring
from mirage.models.fpfh_bank import FpfhBank
from mirage.models.patchcore import PatchCore


def _zscore(maps, valids):
    v = maps[valids]
    return (maps - v.mean()) / (v.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/fused"))
    args = ap.parse_args()

    def fit(c):
        train_rgb = load_split(split="train", label=0, cats=[c], channels=["rgb"], device="cuda")
        pc = PatchCore().fit(train_rgb.x)
        _, train_arr = store.arrays(c, "train", 0)
        fbank = FpfhBank().fit([(a["xyz"], a["valid"]) for a in train_arr])
        return pc, fbank

    def score(state, c):
        pc, fbank = state
        test = load_split(split="test", cats=[c], channels=["rgb"], device="cuda")
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        masks = test.gt.squeeze(1).cpu().numpy().astype(bool)
        pc_maps = pc.score_maps(test.x)
        _, test_arr = store.arrays(c, "test")
        fp_maps = np.stack([fbank.score_map(a["xyz"], a["valid"]) for a in test_arr])
        fused = (_zscore(pc_maps, valids) + _zscore(fp_maps, valids)) * valids
        scores = scoring.image_scores(fused, valids)
        return (fused, valids, masks, scores,
                test.df["label"].to_numpy(), np.array(test.df["defect"].to_list()))

    harness.run("fused_rgb_fpfh", fit, score, cats=args.cats, out=args.out)


if __name__ == "__main__":
    main()
