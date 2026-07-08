"""PatchCore feature memory bank over categories -> aggregate (no training). Uses the eval harness.

Run:  python -m surfscan.experiments.run_patchcore [--cats bagel ...] [--coreset 0.1] [--coreset-method greedy|random]
"""
from __future__ import annotations

import argparse

from core.compute import pick_device
from core.data.dataset import load_split
from core.method import Method
from surfscan.evaluation import harness, scoring
from surfscan.models.patchcore import FitCfg, PatchCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--coreset-method", default="greedy", choices=["greedy", "random"])
    ap.add_argument("--size", type=int, default=None, help="processed store resolution (default 256)")
    args = ap.parse_args()
    dev = pick_device()

    def fit(c):
        train = load_split(split="train", label=0, cats=[c], channels=["rgb"], device=dev, size=args.size)
        return PatchCore(device=dev).fit(train.x, FitCfg(coreset=args.coreset, method=args.coreset_method))

    def score(pc, c):
        test = load_split(split="test", cats=[c], channels=["rgb"], device=dev, size=args.size)
        return scoring.score_arrays(pc.score_maps(test.x), test)

    harness.run(f"patchcore_rgb_{args.coreset_method}", Method(fit, score), cats=args.cats)


if __name__ == "__main__":
    main()
