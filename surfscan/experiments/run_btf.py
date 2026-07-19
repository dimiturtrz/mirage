"""BTF (FPFH geometry memory bank) over categories -> aggregate. The geometry SOTA-floor.

Uses the eval harness. Fit a normal-FPFH bank per category, score the test split. No training — so
`BtfMethod` satisfies `AnomalyMethod` without any Trainer.

Run:  python -m surfscan.run btf [--cats bagel ...] [--size 256]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from core.data.static import store
from core.data.static.mvtec import Split
from core.method import ScoreArrays
from surfscan.evaluation import scoring
from surfscan.method_cli import MethodCli
from surfscan.models.fpfh_bank import FpfhBank


@dataclass(frozen=True)
class BtfCfg:
    size: int | None = field(default=None, metadata={"help": "store resolution (default 256; 512 = native-ish)"})


class BtfMethod:
    run_name = "btf_fpfh"

    def __init__(self, cfg: BtfCfg, dev: torch.device) -> None:
        self.dev = dev
        self.size = cfg.size or 256

    def fit(self, cat: str) -> FpfhBank:
        _, train = store.Store.arrays(cat, Split.TRAIN, 0, self.size)
        return FpfhBank(device=self.dev).fit([(a["xyz"], a["valid"]) for a in train])

    def score(self, state: FpfhBank, cat: str) -> ScoreArrays:
        dft, test = store.Store.arrays(cat, Split.TEST, size=self.size)
        amaps = state.score_maps([(a["xyz"], a["valid"]) for a in test])
        return scoring.Scoring.score_arrays_store(amaps, test, dft)


SPEC = MethodCli.method_spec("btf", BtfMethod, BtfCfg)
