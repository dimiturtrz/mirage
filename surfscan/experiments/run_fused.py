"""Fused PatchCore(rgb) + BTF(FPFH geometry) anomaly maps -> aggregate (M3DM-lite). Uses the harness.

Score each test sample with both banks, z-score each map family over the valid test pixels, sum.

Run:  python -m surfscan.experiments.run_fused [--cats bagel ...]
"""
from __future__ import annotations

import argparse

import numpy as np

from core.compute import pick_device
from core.data import store
from core.data.dataset import load_split
from core.method import Method
from surfscan.evaluation import harness, scoring
from surfscan.models.fpfh_bank import FpfhBank
from surfscan.models.patchcore import PatchCore


def _zscore(maps, valids):
    v = maps[valids]
    return (maps - v.mean()) / (v.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    args = ap.parse_args()
    dev = pick_device()

    def fit(c):
        train_rgb = load_split(split="train", label=0, cats=[c], channels=["rgb"], device=dev)
        pc = PatchCore(device=dev).fit(train_rgb.x)
        _, train_arr = store.arrays(c, "train", 0)
        fbank = FpfhBank(device=dev).fit([(a["xyz"], a["valid"]) for a in train_arr])
        return pc, fbank

    def score(state, c):
        pc, fbank = state
        test = load_split(split="test", cats=[c], channels=["rgb"], device=dev)
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        pc_maps = pc.score_maps(test.x)
        _, test_arr = store.arrays(c, "test")
        fp_maps = np.stack([fbank.score_map(a["xyz"], a["valid"]) for a in test_arr])
        fused = (_zscore(pc_maps, valids) + _zscore(fp_maps, valids)) * valids
        return scoring.score_arrays(fused, test)

    harness.run("fused_rgb_fpfh", Method(fit, score), cats=args.cats)


if __name__ == "__main__":
    main()
