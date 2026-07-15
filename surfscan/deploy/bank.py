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

# Product quantization (arxiv 2603.20288 [S3]): m subspaces, b bits/code -> 2^b centroids per subspace.
_PQ_M = 8
_PQ_BITS = 8
_PQ_IMG_AUROC_COST_PP = 9.0   # measured I-ROC 0.95 -> 0.86 under PQ [S3]

# PatchCore-Lite literature anchor [S3] — the cross-check that the byte formulas reproduce ~334 KB.
_ANCHOR_K = 10_000
_ANCHOR_D = 256


@dataclass(frozen=True)
class BankCost:
    """One bank's storage across representations, and whether each fits the on-chip envelope."""
    label: str
    n_vectors: int
    channels: int
    fp32_mib: float
    int8_mib: float
    pq_kib: float
    fp32_fits_sram: bool
    int8_fits_sram: bool
    pq_fits_sram: bool


class BankMemory:
    """Model the PatchCore bank's storage + host-side search cost against the edge memory envelope."""

    @staticmethod
    def _pq_bytes(n: int, c: int) -> int:
        codes = n * _PQ_M                                   # 1 byte per subspace index (b=8)
        codebook = _PQ_M * (2 ** _PQ_BITS) * (c // _PQ_M) * 4   # fp32 centroids
        return codes + codebook

    @staticmethod
    def cost(label: str, n: int, c: int) -> BankCost:
        envelope = SRAM_CLASS_MIB * _MIB
        fp32, int8, pq = n * c * 4, n * c, BankMemory._pq_bytes(n, c)
        return BankCost(
            label=label, n_vectors=n, channels=c,
            fp32_mib=round(fp32 / _MIB, 2), int8_mib=round(int8 / _MIB, 2), pq_kib=round(pq / 1024, 1),
            fp32_fits_sram=fp32 <= envelope, int8_fits_sram=int8 <= envelope, pq_fits_sram=pq <= envelope)

    @staticmethod
    def host_search_line(n: int, c: int) -> dict:
        """Per-frame bank search: distance-matmul (on-accelerator as Conv1x1) + argmin (host residue)."""
        return {
            "patches_per_frame": _PATCHES_PER_FRAME,
            "distance_macs_g": round(_PATCHES_PER_FRAME * n * c / 1e9, 2),   # Conv1x1 on-accelerator
            "argmin_compares_m": round(_PATCHES_PER_FRAME * n / 1e6, 2),     # over N, HOST-side residue
            "argmin_location": "host",
            "pq_img_auroc_cost_pp": _PQ_IMG_AUROC_COST_PP,
        }

    @staticmethod
    def _log(rows: list[BankCost], search: dict) -> None:
        log.info(f"{'bank':22s} {'N':>7s} {'C':>5s} {'fp32(MiB)':>9s} {'int8(MiB)':>9s} {'PQ(KiB)':>8s} "
                 f"{'fp32<8Mi':>8s} {'int8<8Mi':>8s} {'PQ<8Mi':>6s}")
        for r in rows:
            log.info(f"{r.label:22s} {r.n_vectors:7d} {r.channels:5d} {r.fp32_mib:9.2f} {r.int8_mib:9.2f} "
                     f"{r.pq_kib:8.1f} {r.fp32_fits_sram!s:>8s} {r.int8_fits_sram!s:>8s} "
                     f"{r.pq_fits_sram!s:>6s}")
        log.info(f"host search/frame: {search['distance_macs_g']} GMAC distance (Conv1x1 on-accel) + "
                 f"{search['argmin_compares_m']}M argmin compares (HOST); PQ costs ~{search['pq_img_auroc_cost_pp']}pp img-AUROC")

    @staticmethod
    def section(n: int = _DEFAULT_N, c: int = _DEFAULT_C) -> dict:
        """The bank sub-document the projection embeds: representations + the host-search line."""
        rows = [
            BankMemory.cost("ours", n, c),
            BankMemory.cost("patchcore_lite[S3]", _ANCHOR_K, _ANCHOR_D),   # reproduces ~334 KB PQ / ~9.8 MB fp32
        ]
        return {"sram_class_mib": SRAM_CLASS_MIB, "pq": {"m": _PQ_M, "bits": _PQ_BITS},
                "banks": [asdict(r) for r in rows], "host_search_per_frame": BankMemory.host_search_line(n, c)}

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--n-vectors", type=int, default=_DEFAULT_N, help="bank size (measured from a real fit)")
        ap.add_argument("--channels", type=int, default=_DEFAULT_C, help="feature width C (layer2+layer3)")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        sec = BankMemory.section(args.n_vectors, args.channels)
        BankMemory._log([BankMemory.cost("ours", args.n_vectors, args.channels),
                         BankMemory.cost("patchcore_lite[S3]", _ANCHOR_K, _ANCHOR_D)],
                        sec["host_search_per_frame"])


SPEC = Spec("bank", BankMemory.add_args, BankMemory.run)
