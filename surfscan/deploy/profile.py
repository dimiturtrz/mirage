"""Model profiler — params, inference FLOPs, peak-activation memory, disk footprint per neural model.

The foundation of the deploy cost-model: what each detector actually costs to run one 256x256 frame,
measured (not guessed) with torch's own FlopCounterMode (no new dep). The point is that params and
FLOPs diverge — ConvVAE is param-heavy but compute-light, Draem the opposite — so a single "model
size" number can't rank deployability; you need both, plus the activation peak that sets the SRAM/VRAM
working-set floor. PatchCore's memory-bank cost (coreset x C) is a separate load-bearing line handled
in the bank-memory model (82w.2); here we profile only its backbone forward.

Backbones are profiled TRUNCATED to the layer the detector actually reads (wide_resnet50_2 -> layer2
for feat-recon, -> layer3 for PatchCore); layer4 + fc are never run at inference and are prunable, so
counting them would inflate the deploy footprint. Emits docs/DEPLOY_COST.json (structured source for
the result-table cost columns + the accelerator projection).

    python -m surfscan.deploy profile [--size 256] [--json docs/DEPLOY_COST.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision
from torch.utils.flop_counter import FlopCounterMode

from core.obs import Obs
from surfscan.dispatch import Spec
from surfscan.models.draem import Draem
from surfscan.models.feat_ae import FeatAE
from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import ModelCfg

log = Obs.get()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs" / "DEPLOY_COST.json"
_BYTES_FP32 = 4
_MB = 1e6
_BACKBONE = "wide_resnet50_2"
_CUDA = "cuda"
_LAYER2_CH = 512  # wide_resnet50_2 layer2 output channels (feat-recon AE input depth)


@dataclass(frozen=True)
class CostRow:
    """One model's measured single-frame footprint at the profiled input size."""
    name: str
    params_m: float
    gflops: float
    macs_g: float
    act_mb: float
    disk_fp32_mb: float
    disk_int8_mb: float
    note: str


class TruncatedResnet:
    """A torchvision resnet run only through the layer a detector reads — deploy-relevant sub-network.

    layer4 + fc are never executed at inference (the detector hooks an earlier layer), so both their
    FLOPs and their parameters are excluded — the honest deploy footprint of the frozen backbone.
    """
    _STEM = ("conv1", "bn1", "relu", "maxpool")
    _LAYERS = ("layer1", "layer2", "layer3", "layer4")

    @staticmethod
    def used_names(upto: str) -> list[str]:
        """Module names run at inference: stem + residual stages up to (incl.) `upto`; the rest pruned."""
        return list(TruncatedResnet._STEM) + list(
            TruncatedResnet._LAYERS[: TruncatedResnet._LAYERS.index(upto) + 1])

    def __init__(self, name: str, upto: str, dev: str):
        net = getattr(torchvision.models, name)(weights="DEFAULT").to(dev).eval()
        self.mods = [getattr(net, m) for m in self.used_names(upto)]

    @torch.no_grad()
    def __call__(self, x):
        for m in self.mods:
            x = m(x)
        return x

    @property
    def params(self) -> int:
        return sum(p.numel() for m in self.mods for p in m.parameters())


_, _L2, _L3, _ = TruncatedResnet._LAYERS   # deploy-read depths, bound from the SSOT (no new literals)


class Profiler:
    """Measure params/FLOPs/activation-mem/disk per neural detector and emit the structured cost source."""

    @staticmethod
    def _params(fn) -> int:
        if isinstance(fn, TruncatedResnet):
            return fn.params
        return sum(p.numel() for p in fn.parameters())

    @staticmethod
    def _flops(fn, x) -> int:
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            fn(x)
        return fc.get_total_flops()

    @staticmethod
    def _peak_act_mb(fn, x, dev: str) -> float:
        if dev != _CUDA:
            return float("nan")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            fn(x)
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / _MB

    @staticmethod
    def profile_one(name: str, fn, x, note: str, dev: str) -> CostRow:
        params = Profiler._params(fn)
        flops = Profiler._flops(fn, x)
        return CostRow(
            name=name,
            params_m=round(params / _MB, 2),
            gflops=round(flops / 1e9, 2),
            macs_g=round(flops / 2e9, 2),
            act_mb=round(Profiler._peak_act_mb(fn, x, dev), 1),
            disk_fp32_mb=round(params * _BYTES_FP32 / _MB, 2),
            disk_int8_mb=round(params / _MB, 2),
            note=note,
        )

    @staticmethod
    def targets(size: int, dev: str):
        """The neural detectors + their deploy-relevant input, built on `dev`. FPFH/BTF (open3d, CPU,
        non-neural) is a geometry-bank cost, not a forward-FLOP cost — handled in the bank model."""
        img = torch.randn(1, 3, size, size, device=dev)
        feat = torch.randn(1, _LAYER2_CH, size // 8, size // 8, device=dev)  # layer2 stride-8 feature map
        return [
            ("convvae_xyz", ConvVAE(ModelCfg(in_ch=3)).to(dev).eval(), img, "reconstruction (xyz), full AE"),
            ("draem_rgb", Draem(ch=3).to(dev).eval(), img, "reconstructive+discriminative UNets (rgb)"),
            ("backbone_layer2", TruncatedResnet(_BACKBONE, _L2, dev), img,
             "frozen wide_resnet50_2 -> layer2 (feat-recon extractor; layer3+ pruned)"),
            ("backbone_layer3", TruncatedResnet(_BACKBONE, _L3, dev), img,
             "frozen wide_resnet50_2 -> layer3 (PatchCore extractor; layer4+fc pruned)"),
            ("feat_ae", FeatAE(ch=_LAYER2_CH).to(dev).eval(), feat, "feat-recon head on layer2 feature map"),
        ]

    @staticmethod
    def _log_table(rows: list[CostRow]) -> None:
        log.info(f"{'model':18s} {'params(M)':>10s} {'GFLOPs':>8s} {'MACs(G)':>8s} {'act(MB)':>8s} "
                 f"{'fp32(MB)':>9s} {'int8(MB)':>9s}  note")
        for r in rows:
            log.info(f"{r.name:18s} {r.params_m:10.2f} {r.gflops:8.2f} {r.macs_g:8.2f} {r.act_mb:8.1f} "
                     f"{r.disk_fp32_mb:9.2f} {r.disk_int8_mb:9.2f}  {r.note}")

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--size", type=int, default=256, help="square input side (H=W) to profile at")
        ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="structured cost-source output path")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        dev = _CUDA if torch.cuda.is_available() else "cpu"
        rows = [Profiler.profile_one(name, fn, x, note, dev)
                for name, fn, x, note in Profiler.targets(args.size, dev)]
        Profiler._log_table(rows)
        payload = {"input_size": args.size, "device": dev, "rows": [asdict(r) for r in rows]}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log.info(f"wrote {args.json.relative_to(ROOT)}  ({len(rows)} models @ {args.size}^2, {dev})")


SPEC = Spec("profile", Profiler.add_args, Profiler.run)
