"""PatchCore feature memory bank over categories -> aggregate (no training). Uses the eval harness.

The reference `AnomalyMethod` — a configured object (`PatchCoreCfg`), no bespoke runner: `method_spec`
turns the class + config into the `surfscan.run patchcore` subcommand, and the same class is callable
in code (`PatchCoreMethod(PatchCoreCfg(coreset=0.2), dev).fit("bagel")`).

Run:  python -m surfscan.run patchcore [--cats bagel ...] [--coreset 0.1] [--coreset-method greedy|random]
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.data.dataset import GpuSplit
from surfscan.evaluation import scoring
from surfscan.method_cli import method_spec
from surfscan.models.patchcore import FitCfg, PatchCore


@dataclass(frozen=True)
class PatchCoreCfg:
    coreset: float = 0.1
    coreset_method: str = field(default="greedy", metadata={"choices": ["greedy", "random"]})
    size: int | None = field(default=None, metadata={"help": "processed store resolution (default 256)"})
    amp: bool = field(default=False, metadata={"help": "bf16 backbone forward (faster; off = reported number)"})
    seed: int = field(default=0, metadata={"help": "coreset RNG seed (vary for multi-seed error bars)"})


class PatchCoreMethod:
    """rgb patch-feature memory bank; nearest-neighbour distance to the normal bank = the anomaly map."""

    def __init__(self, cfg: PatchCoreCfg, dev):
        self.cfg = cfg
        self.dev = dev

    @property
    def run_name(self) -> str:
        return f"patchcore_rgb_{self.cfg.coreset_method}"

    def fit(self, cat):
        train = GpuSplit.load_split(split="train", label=0, cats=[cat], channels=["rgb"],
                                    device=self.dev, size=self.cfg.size)
        return PatchCore(device=self.dev, amp=self.cfg.amp).fit(
            train.x, FitCfg(coreset=self.cfg.coreset, method=self.cfg.coreset_method, seed=self.cfg.seed))

    def score(self, state, cat):
        test = GpuSplit.load_split(split="test", cats=[cat], channels=["rgb"], device=self.dev, size=self.cfg.size)
        return scoring.Scoring.score_arrays(state.score_maps(test.x), test)


SPEC = method_spec("patchcore", PatchCoreMethod, PatchCoreCfg)
