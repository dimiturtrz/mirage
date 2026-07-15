"""Memory-bank cost model — a stored kNN bank sized as a NORMAL model, not a special case.

PatchCore (rgb) and BTF (geometry) don't run a forward pass; they store `N` feature vectors of width `C`
and search them per patch. That is still just params + FLOPs + disk, so a bank is emitted as an ordinary
`CostRow` in the model list (`profile`), referenced by a detector in `pieces` like any conv stage:

  params_m     N x C                                        (the stored vectors — a bank's "weights")
  gflops       2 x patches x N x C                          (per-frame distance matmul; a Conv1x1)
  disk_fp32_mb N x C x 4B                                    (the raw store — the honest baseline)
  disk_int8_mb codes N x m + codebook m x 2^b x (C/m) x 4B  (product-quantized edge form — a bank's int8)

PQ is anchored to PatchCore-Lite (arxiv 2603.20288 [S3]): K=10000, d=256, m=8/b=8 -> ~334 KB vs ~9.77 MB
fp32, ~9 pp image-AUROC cost. The argmin/top-k over N is NOT a footprint field — it rides the detector's
op-class (BANK_LOOKUP), which the fit engine maps to a host residue on every commodity NPU.

    python -m surfscan.deploy bank [--n-vectors 20000] [--channels 1536]
"""
from __future__ import annotations

import argparse

from core.obs import Obs
from surfscan.deploy.schema import CostRow
from surfscan.dispatch import Spec

log = Obs.get()

_MB = 1e6

# PatchCore rgb bank: greedy coreset at 0.1 of a pool capped at 200k patches -> 20k vectors; width C =
# layer2 (512) + layer3 (1024) = 1536. Measured: bagel/foam/tire all fit to exactly (20000, 1536).
_DEFAULT_N = 20_000
_DEFAULT_C = 1536
_PATCHES_PER_FRAME = 32 * 32   # layer2 stride-8 grid at 256^2 (the shallowest hooked layer sets the count)

# FPFH geometry bank (BTF / the 3D half of fused). C=33 is fixed by Open3D's FPFH descriptor. N derives from
# the FpfhBank defaults (per_sample=2000, coreset=0.25) over ~244 train views: 244 x 2000 x 0.25 ≈ 122k.
_GEOM_C = 33
_FPFH_PER_SAMPLE = 2_000
_FPFH_CORESET = 0.25
_TRAIN_VIEWS = 244
_GEOM_N = int(_TRAIN_VIEWS * _FPFH_PER_SAMPLE * _FPFH_CORESET)

# Product quantization (arxiv 2603.20288 [S3]): m subspaces, b bits/code -> 2^b centroids per subspace.
_PQ_M = 8
_PQ_BITS = 8

# PatchCore-Lite literature anchor [S3] — the cross-check that the PQ formula reproduces ~334 KB.
_ANCHOR_K = 10_000
_ANCHOR_D = 256

RGB_BANK = "patchcore_rgb_bank"
GEOMETRY_BANK = "fpfh_geometry_bank"


class BankMemory:
    """Size a stored kNN bank as an ordinary model CostRow (params = N x C, flops = the distance matmul)."""

    @staticmethod
    def _pq_bytes(n: int, c: int) -> int:
        codes = n * _PQ_M                                   # 1 byte per subspace index (b=8)
        codebook = _PQ_M * (2 ** _PQ_BITS) * (c // _PQ_M) * 4   # fp32 centroids
        return codes + codebook

    @staticmethod
    def cost(name: str, n: int, c: int, patches: int, note: str) -> CostRow:
        flops = 2 * patches * n * c
        return CostRow(
            name=name, params_m=round(n * c / _MB, 2),
            gflops=round(flops / 1e9, 2), macs_g=round(flops / 2e9, 2),
            act_mb=0.0,                                              # the search is a streamed/chunked reduction
            disk_fp32_mb=round(n * c * 4 / _MB, 2),
            disk_int8_mb=round(BankMemory._pq_bytes(n, c) / _MB, 2),  # product-quantized edge form
            note=note)

    @staticmethod
    def components(n: int = _DEFAULT_N, c: int = _DEFAULT_C) -> list[CostRow]:
        """The bank models a detector can carry as pieces: rgb (PatchCore) + geometry (FPFH/BTF)."""
        return [
            BankMemory.cost(RGB_BANK, n, c, _PATCHES_PER_FRAME,
                            "stored PatchCore rgb feature bank (N x C); int8 col = product-quantized edge form; "
                            "argmin/top-k tail is host-side (op_class bank_lookup)"),
            BankMemory.cost(GEOMETRY_BANK, _GEOM_N, _GEOM_C, _FPFH_PER_SAMPLE,
                            "stored FPFH geometry bank (BTF); handcrafted descriptors, no neural forward; "
                            "int8 col = product-quantized edge form; argmin tail host-side"),
        ]

    @staticmethod
    def _log(rows: list[CostRow]) -> None:
        log.info(f"{'bank':20s} {'N':>7s} {'C':>5s} {'params(M)':>9s} {'GFLOPs':>8s} {'fp32(MB)':>9s} "
                 f"{'PQ int8(MB)':>11s}")
        for r in rows:
            log.info(f"{r.name:20s} {'':>7s} {'':>5s} {r.params_m:9.2f} {r.gflops:8.2f} {r.disk_fp32_mb:9.2f} "
                     f"{r.disk_int8_mb:11.2f}")
        log.info("PQ reproduces the PatchCore-Lite ~334 KB anchor [S3]; the argmin/top-k over N is a HOST "
                 "residue on every commodity NPU (rides op_class bank_lookup).")

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--n-vectors", type=int, default=_DEFAULT_N, help="rgb bank size (measured from a real fit)")
        ap.add_argument("--channels", type=int, default=_DEFAULT_C, help="rgb feature width C (layer2+layer3)")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        BankMemory._log(BankMemory.components(args.n_vectors, args.channels))


SPEC = Spec("bank", BankMemory.add_args, BankMemory.run)
