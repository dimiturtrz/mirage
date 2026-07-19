"""Model profiler — params, inference FLOPs, peak-activation memory, disk footprint per neural model.

The foundation of the deploy fit model: what each detector actually costs to run one 256x256 frame,
measured (not guessed) with torch's own FlopCounterMode (no new dep). The point is that params and
FLOPs diverge — ConvVAE is param-heavy but compute-light, Draem the opposite — so a single "model
size" number can't rank deployability; you need both, plus the activation peak that sets the SRAM/VRAM
working-set floor. PatchCore's memory-bank cost (coreset x C) is a separate load-bearing line handled
in the bank-memory model; here we profile only its backbone forward.

Backbones are profiled TRUNCATED to the layer the detector actually reads (wide_resnet50_2 -> layer2
for feat-recon, -> layer3 for PatchCore); layer4 + fc are never run at inference and are prunable, so
counting them would inflate the deploy footprint.

Emits deploy/models_params.json: the measured `components` (atoms), the `detectors` (shippable
compositions carrying an OP-CLASS — the axis the fit engine matches against accelerator op-support), and
the PatchCore `bank` model. Structured source for the result-table cost columns + the fit matrix.

    python -m surfscan.deploy profile [--size 256]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torchvision
from jaxtyping import Float
from torch import Tensor
from torch.utils.flop_counter import FlopCounterMode

from core.obs import Obs
from surfscan.deploy import MODELS_DOC, bank
from surfscan.deploy.schema import CostRow, OpClass
from surfscan.dispatch import Spec
from surfscan.models.draem import Draem
from surfscan.models.feat_ae import FeatAE
from surfscan.models.pointmae import Pointmae
from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import ModelCfg

log = Obs.get()

_BYTES_FP32 = 4
_MB = 1e6
_BACKBONE = "wide_resnet50_2"
_CUDA = "cuda"
_LAYER2_CH = 512  # wide_resnet50_2 layer2 output channels (feat-recon AE input depth)
_POINTMAE = "pointmae"
_POINTMAE_N = 2048  # points fed to the Point-MAE transformer (num_group 1024 x group_size 128 -> grouped internally)
_OPTIONAL_DEP = (ImportError, FileNotFoundError, RuntimeError, OSError)  # external/M3DM may be absent


@dataclass(frozen=True)
class Detector:
    """A shippable detector as a composition of components + its op-class (the fit axis).

    `pieces` are `components` names summed for the footprint — conv stages AND stored kNN banks alike (a
    bank is just another component). The argmin/top-k tail rides `op_class` (BANK_LOOKUP), not a piece."""
    name: str
    op_class: OpClass
    pieces: tuple[str, ...]


class TruncatedResnet:
    """A torchvision resnet run only through the layer a detector reads — deploy-relevant sub-network.

    layer4 + fc are never executed at inference (the detector hooks an earlier layer), so both their
    FLOPs and their parameters are excluded — the honest deploy footprint of the frozen backbone.
    """
    _STEM = ("conv1", "bn1", "relu", "maxpool")
    LAYERS = ("layer1", "layer2", "layer3", "layer4")

    @staticmethod
    def used_names(upto: str) -> list[str]:
        """Module names run at inference: stem + residual stages up to (incl.) `upto`; the rest pruned."""
        return list(TruncatedResnet._STEM) + list(
            TruncatedResnet.LAYERS[: TruncatedResnet.LAYERS.index(upto) + 1])

    def __init__(self, name: str, upto: str, dev: str):
        net = getattr(torchvision.models, name)(weights="DEFAULT").to(dev).eval()
        self.mods = [getattr(net, m) for m in self.used_names(upto)]

    @torch.no_grad()
    def __call__(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        for m in self.mods:
            x = m(x)
        return x

    def export_spec(self, x: Float[Tensor, "b c h w"]) -> tuple[torch.nn.Module, Float[Tensor, "b c h w"]]:
        """The exportable nn.Module + its input — the truncated stages as a Sequential (ONNX gate)."""
        return torch.nn.Sequential(*self.mods), x

    @property
    def params(self) -> int:
        return sum(p.numel() for m in self.mods for p in m.parameters())


_, _L2, _L3, _ = TruncatedResnet.LAYERS   # deploy-read depths, bound from the SSOT (no new literals)


class PointmaeBackbone:
    """M3DM's Point-MAE point-transformer as a profile target — the real transformer_attention op-class.

    Architecture only (ckpt=None): FLOP/param counts don't need trained weights. Optional dependency —
    needs external/M3DM checked out; `targets()` skips it if the import/load raises.
    """
    def __init__(self, dev: str):
        self.net = Pointmae.load_pointmae(dev, ckpt=None)

    @torch.no_grad()
    def __call__(self, x: Float[Tensor, "b n c"]) -> tuple[Float[Tensor, "b g c"], Float[Tensor, "b g 3"]]:
        return Pointmae.pointmae_features(self.net, x)

    def export_spec(self, x: Float[Tensor, "b n c"]) -> tuple[Any, Float[Tensor, "b c n"]]:
        """The transformer module + its (B,3,N) input — the Point-MAE forward wants channels-first."""
        return self.net, x.transpose(1, 2).contiguous()

    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


ProfileTarget = ConvVAE | Draem | FeatAE | TruncatedResnet | PointmaeBackbone


class Profiler:
    """Measure params/FLOPs/activation-mem/disk per neural detector and emit the structured cost source."""

    _DETECTORS = (
        Detector("convvae", OpClass.CONV_NATIVE, ("convvae_xyz",)),
        Detector("draem", OpClass.CONV_NATIVE, ("draem_rgb",)),
        Detector("feat_recon", OpClass.CONV_NATIVE, ("backbone_layer2", "feat_ae")),
        Detector("patchcore", OpClass.BANK_LOOKUP, ("backbone_layer3", bank.RGB_BANK)),
        Detector("btf", OpClass.BANK_LOOKUP, (bank.GEOMETRY_BANK,)),                    # geometry bank only
        Detector("fused", OpClass.BANK_LOOKUP, ("backbone_layer3", bank.RGB_BANK, bank.GEOMETRY_BANK)),
        Detector(_POINTMAE, OpClass.TRANSFORMER_ATTENTION, (_POINTMAE,)),
    )

    @staticmethod
    def _params(fn: Any) -> int:
        own = getattr(fn, "params", None)   # profile wrappers expose .params; a bare nn.Module does not
        return own if own is not None else sum(p.numel() for p in fn.parameters())

    @staticmethod
    def _flops(fn: Any, x: Float[Tensor, "b *shape"]) -> int:
        fc = FlopCounterMode(display=False)
        with torch.no_grad(), fc:
            fn(x)
        return fc.get_total_flops()

    @staticmethod
    def _peak_act_mb(fn: Any, x: Float[Tensor, "b *shape"], dev: str) -> float:
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
    def profile_one(name: str, fn: Any, x: Float[Tensor, "b *shape"], note: str, dev: str) -> CostRow:
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
    def targets(size: int, dev: str) -> list[tuple[str, ProfileTarget, Float[Tensor, "b *shape"], str]]:
        """The neural detectors + their deploy-relevant input, built on `dev`. FPFH/BTF (open3d, CPU,
        non-neural) is a geometry-bank cost, not a forward-FLOP cost — handled in the bank model."""
        img = torch.randn(1, 3, size, size, device=dev)
        feat = torch.randn(1, _LAYER2_CH, size // 8, size // 8, device=dev)  # layer2 stride-8 feature map
        out: list[tuple[str, ProfileTarget, Float[Tensor, "b *shape"], str]] = [
            ("convvae_xyz", ConvVAE(ModelCfg(in_ch=3)).to(dev).eval(), img, "reconstruction (xyz), full AE"),
            ("draem_rgb", Draem(ch=3).to(dev).eval(), img, "reconstructive+discriminative UNets (rgb)"),
            ("backbone_layer2", TruncatedResnet(_BACKBONE, _L2, dev), img,
             "frozen wide_resnet50_2 -> layer2 (feat-recon extractor; layer3+ pruned)"),
            ("backbone_layer3", TruncatedResnet(_BACKBONE, _L3, dev), img,
             "frozen wide_resnet50_2 -> layer3 (PatchCore extractor; layer4+fc pruned)"),
            ("feat_ae", FeatAE(ch=_LAYER2_CH).to(dev).eval(), feat, "feat-recon head on layer2 feature map"),
        ]
        pts = torch.randn(1, _POINTMAE_N, 3, device=dev)
        try:
            out.append((_POINTMAE, PointmaeBackbone(dev), pts,
                        "M3DM Point-MAE point-transformer (attention ops; xyz points, not pixels)"))
        except _OPTIONAL_DEP:
            log.info("pointmae skipped — external/M3DM not available (transformer op-class stays modelled, unprofiled)")
        return out

    @staticmethod
    def _log_table(rows: list[CostRow]) -> None:
        log.info(f"{'model':18s} {'params(M)':>10s} {'GFLOPs':>8s} {'MACs(G)':>8s} {'act(MB)':>8s} "
                 f"{'fp32(MB)':>9s} {'int8(MB)':>9s}  note")
        for r in rows:
            log.info(f"{r.name:18s} {r.params_m:10.2f} {r.gflops:8.2f} {r.macs_g:8.2f} {r.act_mb:8.1f} "
                     f"{r.disk_fp32_mb:9.2f} {r.disk_int8_mb:9.2f}  {r.note}")

    @staticmethod
    def device() -> str:
        return _CUDA if torch.cuda.is_available() else "cpu"

    @staticmethod
    def measure(size: int, dev: str) -> list[CostRow]:
        """Measure every neural component's footprint in-memory — the fit matrix embeds these directly."""
        return [Profiler.profile_one(name, fn, x, note, dev)
                for name, fn, x, note in Profiler.targets(size, dev)]

    @staticmethod
    def document(size: int, dev: str) -> dict[str, Any]:
        """The deploy/models_params.json payload: components (neural forwards AND banks, one list) + detectors.

        A bank is a normal component — computed analytically (N x C), not FlopCounterMode-measured — so it
        joins the neural rows in `components` and detectors reference it in `pieces` like any conv stage."""
        components = Profiler.measure(size, dev) + bank.BankMemory.components()
        have = {r.name for r in components}
        detectors = [d for d in Profiler._DETECTORS if all(p in have for p in d.pieces)]  # drop an unprofiled optional
        return {
            "note": "generated by `python -m surfscan.deploy profile` — single-frame footprint "
                    "(banks sized analytically)",
            "input_size": size, "device": dev,
            "components": [asdict(r) for r in components],
            "detectors": [{"name": d.name, "op_class": d.op_class, "pieces": list(d.pieces)} for d in detectors],
        }

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--size", type=int, default=256, help="square input side (H=W) to profile at")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        dev = Profiler.device()
        doc = Profiler.document(args.size, dev)
        Profiler._log_table([CostRow(**c) for c in doc["components"]])
        MODELS_DOC.parent.mkdir(parents=True, exist_ok=True)
        MODELS_DOC.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        log.info(f"wrote {MODELS_DOC.name}  ({len(doc['components'])} components, "
                 f"{len(doc['detectors'])} detectors)")


SPEC = Spec("profile", Profiler.add_args, Profiler.run)
