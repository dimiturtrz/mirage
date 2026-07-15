"""PatchCore memory-bank cost model — the load-bearing deploy constraint is bank MEMORY, not FLOPs.

A conv detector's cost is its forward pass (profiled in `profile`). PatchCore adds a stored bank of
`N` feature vectors of width `C` that must live somewhere and be searched per patch — and THAT, not the
backbone FLOPs, is what breaks the ~8 MiB on-chip envelope. This models the bank three ways against the
envelope:

  fp32   N x C x 4B                         — as fit; the honest baseline, and what blows the budget
  int8   N x C x 1B                         — 4x smaller, still not on a fixed-function NPU's int8 graph
  int8+PQ  codes N x m  +  codebook m x 2^b x (C/m) x 4B   — product-quantized; fits comfortably

Anchored to the PatchCore-Lite result (arxiv 2603.20288 [S3]): K=10000, d=256, m=8/b=8 -> ~334 KB PQ
vs ~9.77 MB fp32, at a measured ~9 pp image-AUROC cost (I-ROC 0.95 -> 0.86; localization survives,
P-ROC 0.96 -> 0.94). Our bank is wider (C=1536 = layer2 512 + layer3 1024), so its fp32 footprint is
proportionally larger.

The search itself splits: the distance field reduces to a frozen Conv1x1 (W=-2B, bias=||b||^2) that
runs on-accelerator, but the argmin/top-k over the N bank dimension is a HOST-side residue on every
commodity NPU (Coral/Hailo: not in op graph; RKNN: ArgMin yes, TopK no). Reported as a separate
host-side cost line — it is not on-accelerator work.

    python -m surfscan.deploy bank [--n-vectors 20000] [--channels 1536]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

from core.obs import Obs
from surfscan.deploy.accelerators import SRAM_CLASS_MIB
from surfscan.dispatch import Spec

log = Obs.get()

_MIB = 1024 ** 2

# PatchCore config: greedy coreset at 0.1 of a pool capped at 200k patches -> 20k vectors at the cap;
# width C = layer2 (512) + layer3 (1024) = 1536. Measured: bagel/foam/tire all fit to exactly (20000,
# 1536) — every category's ~210-244 train views x 1024 layer2 patches overruns the 200k pool cap.
_DEFAULT_N = 20_000
_DEFAULT_C = 1536
_PATCHES_PER_FRAME = 32 * 32   # layer2 stride-8 grid at 256^2 (the shallowest hooked layer sets the count)

# FPFH geometry bank (BTF / the 3D half of fused). C=33 is fixed by Open3D's FPFH descriptor. N derives from
# the FpfhBank defaults (per_sample=2000, coreset=0.25) over ~244 train views: 244 x 2000 x 0.25 ≈ 122k
# vectors. Per-frame search compares ~per_sample=2000 test descriptors against the bank (its own host tail).
_GEOM_C = 33
_FPFH_PER_SAMPLE = 2_000
_FPFH_CORESET = 0.25
_TRAIN_VIEWS = 244
_GEOM_N = int(_TRAIN_VIEWS * _FPFH_PER_SAMPLE * _FPFH_CORESET)

# Product quantization (arxiv 2603.20288 [S3]): m subspaces, b bits/code -> 2^b centroids per subspace.
_PQ_M = 8
_PQ_BITS = 8
_PQ_IMG_AUROC_COST_PP = 9.0   # measured I-ROC 0.95 -> 0.86 under PQ [S3]

# PatchCore-Lite literature anchor [S3] — the cross-check that the byte formulas reproduce ~334 KB.
_ANCHOR_K = 10_000
_ANCHOR_D = 256


@dataclass(frozen=True)
class BankCost:
    """One bank's storage across representations + its per-frame search cost, vs the on-chip envelope."""
    label: str
    n_vectors: int
    channels: int
    patches_per_frame: int
    fp32_mib: float
    int8_mib: float
    pq_kib: float
    distance_macs_g: float    # per-frame distance matmul (on-accelerator as Conv1x1)
    argmin_compares_m: float  # per-frame argmin over N — the HOST-side residue
    fp32_fits_sram: bool
    int8_fits_sram: bool
    pq_fits_sram: bool


class BankMemory:
    """Model each memory bank's storage + host-side search cost against the edge memory envelope."""

    @staticmethod
    def _pq_bytes(n: int, c: int) -> int:
        codes = n * _PQ_M                                   # 1 byte per subspace index (b=8)
        codebook = _PQ_M * (2 ** _PQ_BITS) * (c // _PQ_M) * 4   # fp32 centroids
        return codes + codebook

    @staticmethod
    def cost(label: str, n: int, c: int, patches: int = _PATCHES_PER_FRAME) -> BankCost:
        envelope = SRAM_CLASS_MIB * _MIB
        fp32, int8, pq = n * c * 4, n * c, BankMemory._pq_bytes(n, c)
        return BankCost(
            label=label, n_vectors=n, channels=c, patches_per_frame=patches,
            fp32_mib=round(fp32 / _MIB, 2), int8_mib=round(int8 / _MIB, 2), pq_kib=round(pq / 1024, 1),
            distance_macs_g=round(patches * n * c / 1e9, 2),   # Conv1x1 on-accelerator
            argmin_compares_m=round(patches * n / 1e6, 2),     # over N, HOST-side residue
            fp32_fits_sram=fp32 <= envelope, int8_fits_sram=int8 <= envelope, pq_fits_sram=pq <= envelope)

    @staticmethod
    def _log(banks: dict[str, BankCost]) -> None:
        log.info(f"{'role':10s} {'bank':18s} {'N':>7s} {'C':>5s} {'fp32(MiB)':>9s} {'int8(MiB)':>9s} "
                 f"{'PQ(KiB)':>8s} {'dist(GMAC)':>10s} {'argmin(M)':>9s} {'PQ<8Mi':>6s}")
        for role, r in banks.items():
            log.info(f"{role:10s} {r.label:18s} {r.n_vectors:7d} {r.channels:5d} {r.fp32_mib:9.2f} "
                     f"{r.int8_mib:9.2f} {r.pq_kib:8.1f} {r.distance_macs_g:10.2f} {r.argmin_compares_m:9.2f} "
                     f"{r.pq_fits_sram!s:>6s}")
        log.info(f"the distance matmul is a Conv1x1 (on-accelerator); the argmin/top-k over N is a HOST residue "
                 f"on every commodity NPU. PQ costs ~{_PQ_IMG_AUROC_COST_PP}pp img-AUROC [S3].")

    @staticmethod
    def roles(n: int = _DEFAULT_N, c: int = _DEFAULT_C) -> dict[str, BankCost]:
        """The deployable banks a detector can carry, keyed by role: rgb (PatchCore) + geometry (FPFH/BTF)."""
        return {"rgb": BankMemory.cost("rgb_patchcore", n, c),
                "geometry": BankMemory.cost("geometry_fpfh", _GEOM_N, _GEOM_C, _FPFH_PER_SAMPLE)}

    @staticmethod
    def section(n: int = _DEFAULT_N, c: int = _DEFAULT_C) -> dict:
        """The bank sub-document the models_params doc embeds: banks-by-role + the PatchCore-Lite anchor."""
        banks = BankMemory.roles(n, c)
        anchor = BankMemory.cost("patchcore_lite[S3]", _ANCHOR_K, _ANCHOR_D)  # reproduces ~334 KB PQ / ~9.8 MB fp32
        return {"sram_class_mib": SRAM_CLASS_MIB, "pq": {"m": _PQ_M, "bits": _PQ_BITS},
                "pq_img_auroc_cost_pp": _PQ_IMG_AUROC_COST_PP,
                "banks": {role: asdict(b) for role, b in banks.items()},
                "anchor": asdict(anchor)}

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--n-vectors", type=int, default=_DEFAULT_N, help="rgb bank size (measured from a real fit)")
        ap.add_argument("--channels", type=int, default=_DEFAULT_C, help="rgb feature width C (layer2+layer3)")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        BankMemory._log(BankMemory.roles(args.n_vectors, args.channels))


SPEC = Spec("bank", BankMemory.add_args, BankMemory.run)
