"""Fused PatchCore(rgb) + BTF(FPFH geometry) anomaly maps -> aggregate (M3DM-lite). Uses the harness.

Score each test sample with both banks, z-score each map family over the valid test pixels, sum. No
training — two banks fit, then fused at score time.

Run:  python -m surfscan.run fused [--cats bagel ...]
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.data import store
from core.data.dataset import GpuSplit
from surfscan.evaluation import scoring
from surfscan.method_cli import method_spec
from surfscan.models.fpfh_bank import FpfhBank
from surfscan.models.patchcore import PatchCore


@dataclass(frozen=True)
class FusedCfg:
    amp: bool = field(default=False, metadata={"help": "bf16 PatchCore backbone forward (faster; off = reported)"})


class FusedMethod:
    run_name = "fused_rgb_fpfh"

    def __init__(self, cfg: FusedCfg, dev):
        self.cfg = cfg
        self.dev = dev

    @staticmethod
    def _zscore(maps, valids):
        v = maps[valids]
        return (maps - v.mean()) / (v.std() + 1e-9)

    def fit(self, cat):
        train_rgb = GpuSplit.load_split(split="train", label=0, cats=[cat], channels=["rgb"], device=self.dev)
        pc = PatchCore(device=self.dev, amp=self.cfg.amp).fit(train_rgb.x)
        _, train_arr = store.Store.arrays(cat, "train", 0)
        fbank = FpfhBank(device=self.dev).fit([(a["xyz"], a["valid"]) for a in train_arr])
        return pc, fbank

    def score(self, state, cat):
        pc, fbank = state
        test = GpuSplit.load_split(split="test", cats=[cat], channels=["rgb"], device=self.dev)
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        pc_maps = pc.score_maps(test.x)
        _, test_arr = store.Store.arrays(cat, "test")
        fp_maps = fbank.score_maps([(a["xyz"], a["valid"]) for a in test_arr])
        fused = (FusedMethod._zscore(pc_maps, valids) + FusedMethod._zscore(fp_maps, valids)) * valids
        return scoring.Scoring.score_arrays(fused, test)


SPEC = method_spec("fused", FusedMethod, FusedCfg)
