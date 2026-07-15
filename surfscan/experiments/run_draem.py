"""DRAEM over categories -> aggregate. Synthesis-recon, Stage-0, self-contained. Uses the harness.

Per category: train recon+disc on good samples with synthetic proxy anomalies, score the test
split by the discriminator output. Default channel = xyz (the geometry signal). The training loop is
the shared `Trainer`; the method supplies the per-batch synth+forward+loss as its step_fn.

Run:  python -m surfscan.run draem [--cats bagel ...] [--epochs 150] [--channels xyz]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from core.compute import Compute
from core.data.dataset import GpuSplit
from core.data.defects import Defects  # channel-aware coherent defects
from core.data.mvtec import Split
from surfscan.evaluation import scoring
from surfscan.method_cli import MethodCli
from surfscan.models.draem import Draem, DraemSynth
from surfscan.training.trainer import Trainer


@dataclass(frozen=True)
class DraemCfg:
    channels: list[str] = field(default_factory=lambda: ["xyz"])
    epochs: int = 150
    seed: int = 0
    synth: str = field(default="realistic", metadata={"choices": ["realistic", "perlin"]})


class DraemMethod:
    def __init__(self, cfg: DraemCfg, dev):
        self.cfg = cfg
        self.dev = dev

    @property
    def run_name(self) -> str:
        return f"draem_{self.cfg.synth}_{'_'.join(self.cfg.channels)}"

    def fit(self, cat):
        torch.manual_seed(self.cfg.seed)
        rng = np.random.RandomState(self.cfg.seed)
        train = GpuSplit.load_split(split=Split.TRAIN, label=0, cats=[cat], channels=self.cfg.channels, device=self.dev)
        model = Draem(ch=train.in_ch).to(self.dev, memory_format=torch.channels_last)
        opt = optim.AdamW(model.parameters(), lr=1e-3)

        def step(idx):
            x, v = train.x[idx], train.valid[idx]
            aug, mask = (Defects(rng).synthesize(x, v, channels=self.cfg.channels)
                         if self.cfg.synth == "realistic" else DraemSynth(rng).synthesize(x, v))
            with Compute.autocast(x):
                rec, logits = model(aug.to(memory_format=torch.channels_last))
                rl = (((rec - x) ** 2) * v).sum() / (v.sum() * x.shape[1] + 1e-8)
                dl = F.binary_cross_entropy_with_logits(logits.float(), mask, weight=v)
                return rl + dl

        return Trainer(model, opt, self.dev, batch=16).fit(len(train), self.cfg.epochs, step)

    def score(self, model, cat):
        test = GpuSplit.load_split(split=Split.TEST, cats=[cat], channels=self.cfg.channels, device=self.dev)
        model.eval()
        amaps = []
        with torch.no_grad():
            for i in range(0, len(test), 16):
                x, v = test.x[i:i + 16], test.valid[i:i + 16]
                _, logits = model(x.to(memory_format=torch.channels_last))
                amaps.append((torch.sigmoid(logits.float()) * v).squeeze(1).cpu())
        return scoring.Scoring.score_arrays(torch.cat(amaps).numpy(), test)


SPEC = MethodCli.method_spec("draem", DraemMethod, DraemCfg)
