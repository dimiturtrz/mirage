"""Fused PatchCore(rgb) + BTF(FPFH geometry) anomaly maps -> aggregate (M3DM-lite). Uses the harness.

Score each test sample with both banks, z-score each map family over the valid test pixels, weighted
sum (rgb_weight; geo down-weighted — equal-sum sits in the au_pro valley). No training — two banks
fit, then fused at score time.

Run:  python -m surfscan.run fused [--cats bagel ...]
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.data.static import store
from core.data.static.dataset import GpuSplit
from core.data.static.mvtec import Split
from surfscan.evaluation import scoring
from surfscan.method_cli import MethodCli
from surfscan.models.fpfh_bank import FpfhBank
from surfscan.models.patchcore import PatchCore


@dataclass(frozen=True)
class FusedCfg:
    amp: bool = field(default=False, metadata={"help": "bf16 PatchCore backbone forward (faster; off = reported)"})
    rgb_weight: float = field(default=0.7, metadata={"help": "fusion weight on rgb PatchCore vs geo FPFH "
                                                            "(1-w). 0.7 = down-weight the weaker geo bank ~2:1; "
                                                            "held-out au_pro peak (equal-sum 0.5 sits in the valley)"})


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
        train_rgb = GpuSplit.load_split(split=Split.TRAIN, label=0, cats=[cat], channels=["rgb"], device=self.dev)
        pc = PatchCore(device=self.dev, amp=self.cfg.amp).fit(train_rgb.x)
        _, train_arr = store.Store.arrays(cat, Split.TRAIN, 0)
        fbank = FpfhBank(device=self.dev).fit([(a["xyz"], a["valid"]) for a in train_arr])
        return pc, fbank

    def score(self, state, cat):
        pc, fbank = state
        test = GpuSplit.load_split(split=Split.TEST, cats=[cat], channels=["rgb"], device=self.dev)
        valids = test.valid.squeeze(1).cpu().numpy().astype(bool)
        pc_maps = pc.score_maps(test.x)
        _, test_arr = store.Store.arrays(cat, Split.TEST)
        fp_maps = fbank.score_maps([(a["xyz"], a["valid"]) for a in test_arr])
        w = self.cfg.rgb_weight
        fused = (w * FusedMethod._zscore(pc_maps, valids) + (1 - w) * FusedMethod._zscore(fp_maps, valids)) * valids
        return scoring.Scoring.score_arrays(fused, test)


SPEC = MethodCli.method_spec("fused", FusedMethod, FusedCfg)
