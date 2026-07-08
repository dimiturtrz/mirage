"""PatchCore feature memory bank over categories -> aggregate (no training). Uses the eval harness.

Run:  python -m surfscan.run patchcore [--cats bagel ...] [--coreset 0.1] [--coreset-method greedy|random]
"""
from __future__ import annotations

from core.compute import pick_device
from core.data.dataset import load_split
from core.method import Method
from surfscan.dispatch import Spec, add_cats
from surfscan.evaluation import harness, scoring
from surfscan.models.patchcore import FitCfg, PatchCore


def _args(ap):
    add_cats(ap)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--coreset-method", default="greedy", choices=["greedy", "random"])
    ap.add_argument("--size", type=int, default=None, help="processed store resolution (default 256)")


def _run(args):
    dev = pick_device()

    def fit(c):
        train = load_split(split="train", label=0, cats=[c], channels=["rgb"], device=dev, size=args.size)
        return PatchCore(device=dev).fit(train.x, FitCfg(coreset=args.coreset, method=args.coreset_method))

    def score(pc, c):
        test = load_split(split="test", cats=[c], channels=["rgb"], device=dev, size=args.size)
        return scoring.score_arrays(pc.score_maps(test.x), test)

    harness.run(f"patchcore_rgb_{args.coreset_method}", Method(fit, score), cats=args.cats)


SPEC = Spec("patchcore", _args, _run)
