"""PatchCore feature memory bank over categories -> aggregate (no training). Uses the eval harness.

Run:  python -m surfscan.experiments.run_patchcore [--cats bagel ...] [--coreset 0.1] [--coreset-method greedy|random]
"""
from __future__ import annotations

import argparse

import numpy as np

from surfscan.data.dataset import load_split
from surfscan.evaluation import harness, scoring
from surfscan.models.patchcore import PatchCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--coreset-method", default="greedy", choices=["greedy", "random"])
    ap.add_argument("--size", type=int, default=None, help="processed store resolution (default 256)")
    args = ap.parse_args()

    def fit(c):
        train = load_split(split="train", label=0, cats=[c], channels=["rgb"], device="cuda", size=args.size)
        return PatchCore().fit(train.x, coreset=args.coreset, method=args.coreset_method)

    def score(pc, c):
        test = load_split(split="test", cats=[c], channels=["rgb"], device="cuda", size=args.size)
        amaps = pc.score_maps(test.x)
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        masks = test.gt.squeeze(1).cpu().numpy().astype(bool)
        scores = scoring.image_scores(amaps, valids)
        return (amaps, valids, masks, scores,
                test.df["label"].to_numpy(), np.array(test.df["defect"].to_list()))

    harness.run(f"patchcore_rgb_{args.coreset_method}", fit, score, cats=args.cats)


if __name__ == "__main__":
    main()
