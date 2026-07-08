"""BTF (FPFH geometry memory bank) over categories -> aggregate. The geometry SOTA-floor.

Uses the eval harness. Fit a normal-FPFH bank per category, score the test split. No training.

Run:  python -m surfscan.experiments.run_btf [--cats bagel ...]
"""
from __future__ import annotations

import argparse

import numpy as np

from core.data import store
from core.method import Method
from surfscan.evaluation import harness, scoring
from surfscan.models.fpfh_bank import FpfhBank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--size", type=int, default=None, help="store resolution (default 256; 512 = native-ish)")
    args = ap.parse_args()
    size = args.size or 256

    def fit(c):
        _, train = store.arrays(c, "train", 0, size)
        return FpfhBank().fit([(a["xyz"], a["valid"]) for a in train])

    def score(bank, c):
        dft, test = store.arrays(c, "test", size=size)
        amaps = np.stack([bank.score_map(a["xyz"], a["valid"]) for a in test])
        valids = np.stack([a["valid"].astype(bool) for a in test])
        masks = np.stack([(a["gt"] > 0) for a in test])
        scores = scoring.image_scores(amaps, valids)
        return (amaps, valids, masks, scores,
                dft["label"].to_numpy(), np.array(dft["defect"].to_list()))

    harness.run("btf_fpfh", Method(fit, score), cats=args.cats)


if __name__ == "__main__":
    main()
