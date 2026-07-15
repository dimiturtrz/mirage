"""Deploy projection — join measured model cost x accelerator specs into a per-pair deployability verdict.

The synthesis step: reads the measured footprints (MODEL_FOOTPRINT.json), the bank model (BANK_MEMORY.json),
and the cited accelerator specs (ACCELERATOR_SPECS.json), and for each (detector x accelerator) pair reports
whether it fits the on-chip envelope, a projected-latency BAND, and a memory-utilization %.

INTEGRITY BOUNDARY: this is a projection, not a measurement. Latency is FLOPs / peak-TOPS scaled by a
stated hardware-efficiency band (30-70%) — a range, never a single measured FPS. It is reported only
where a compute rate is actually in the source; where TOPS is unquoted the pair carries a memory/op-fit
verdict but no latency. The op-fit verdict is the load-bearing output: a conv autoencoder quantizes and
fits everywhere; PatchCore's backbone + distance-matmul run on-accelerator but its argmin/top-k tail is
a HOST residue on every commodity NPU, so PatchCore is on-device only on Jetson.

    python -m surfscan.deploy project [--target-fps 30]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from core.obs import Obs
from surfscan.deploy import DOCS, accelerators, bank, profile
from surfscan.deploy.accelerators import Accelerator
from surfscan.dispatch import Spec

log = Obs.get()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = DOCS / "DEPLOY_PROJECTION.json"
_MIB = 1024 ** 2
_EFF_LO, _EFF_HI = 0.30, 0.70   # sustained fraction of peak TOPS (dataflow NPUs rarely exceed ~70% [roofline])
_TARGET_FPS = 30.0              # industrial inline budget the compute-utilization % is stated against


_DRAM_FREE = "DRAM-free"   # ext_memory marker: all memory on-die, no off-chip streaming fallback


class Verdict(StrEnum):
    """Per-pair op-support deployability — whether the compute graph runs on-accelerator or spills to host."""
    FITS_ONDEVICE = "fits-ondevice"     # conv AE: quantizes + runs fully on-accel, no host tail
    BACKBONE_ONLY = "backbone+host-tail"  # PatchCore on NPU: backbone/matmul on-accel, argmin/top-k on host
    FULL_ONDEVICE = "full-ondevice"     # PatchCore on Jetson: lookup native


class OnchipFit(StrEnum):
    """Whether the working set fits on-chip — separate from op-support (a big model still runs, streamed)."""
    FITS = "fits"                       # working set <= on-chip envelope
    STREAMS_OFFCHIP = "streams-offchip"  # exceeds on-chip but off-chip host RAM exists (latency penalty)
    OVER_ONDIE = "over-ondie"           # exceeds a DRAM-free on-die envelope — no fallback, does not fit
    UNKNOWN = "unknown"                 # on-chip capacity not in the source (Hailo portal)


@dataclass(frozen=True)
class Detector:
    """A shippable detector as a composition of profiled pieces (+ a memory bank for PatchCore-family)."""
    name: str
    pieces: tuple[str, ...]     # profile row names summed for compute; max-act for the working set
    has_bank: bool              # a stored kNN bank whose argmin/top-k is a host residue on commodity NPUs


@dataclass(frozen=True)
class ProjRow:
    """One (detector x accelerator) projection. `None` fields = the source lacks the spec to compute them."""
    detector: str
    accelerator: str
    gflops: float
    work_mem_mib: float
    envelope_mib: float | None
    mem_util_pct: float | None
    onchip_fit: OnchipFit
    latency_ms_lo: float | None
    latency_ms_hi: float | None
    fps_lo: float | None
    fps_hi: float | None
    compute_util_pct_hi: float | None
    verdict: Verdict


class Projection:
    """Join measured cost x accelerator specs into per-pair fits-memory / latency-band / util verdicts."""

    _DETECTORS = (
        Detector("convvae", ("convvae_xyz",), has_bank=False),
        Detector("draem", ("draem_rgb",), has_bank=False),
        Detector("feat_recon", ("backbone_layer2", "feat_ae"), has_bank=False),
        Detector("patchcore", ("backbone_layer3",), has_bank=True),
    )

    @staticmethod
    def _load_cost(path: Path) -> dict:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {r["name"]: r for r in data["rows"]}

    @staticmethod
    def _verdict(det: Detector, acc: Accelerator) -> Verdict:
        if not det.has_bank:
            return Verdict.FITS_ONDEVICE
        return Verdict.FULL_ONDEVICE if acc.patchcore_lookup_ondevice else Verdict.BACKBONE_ONLY

    @staticmethod
    def _onchip_fit(work_mib: float, acc: Accelerator) -> OnchipFit:
        if acc.sram_mib is None:
            return OnchipFit.UNKNOWN
        if work_mib <= acc.sram_mib:
            return OnchipFit.FITS
        return OnchipFit.OVER_ONDIE if _DRAM_FREE in acc.ext_memory else OnchipFit.STREAMS_OFFCHIP

    @staticmethod
    def _bank_mib(det: Detector, bank_cost: dict) -> float:
        return bank_cost["banks"][0]["pq_kib"] / 1024 if det.has_bank else 0.0   # PQ = the edge-deploy bank form

    @staticmethod
    def _compute_cost(det: Detector, rows: dict, bank_cost: dict) -> tuple[float, float]:
        """(summed GFLOPs, working-set MiB) for the detector: conv forward(s) + bank distance-matmul."""
        gflops = sum(rows[p]["gflops"] for p in det.pieces)
        model_mib = sum(rows[p]["disk_int8_mb"] * 1e6 / _MIB for p in det.pieces)   # int8 weights (MB->MiB)
        act_mib = max(rows[p]["act_mb"] * 1e6 / _MIB for p in det.pieces)           # peak activation, one piece at a time
        if det.has_bank:
            gflops += bank_cost["host_search_per_frame"]["distance_macs_g"] * 2     # Conv1x1 MACs -> FLOPs
        return gflops, model_mib + act_mib + Projection._bank_mib(det, bank_cost)

    @staticmethod
    def _latency(gflops: float, acc: Accelerator):
        """Projected latency band (ms) + FPS band from FLOPs / (peak-TOPS x efficiency). None if no TOPS."""
        if acc.int8_tops is None:
            return None, None, None, None, None
        tflops = gflops / 1e3
        ms_lo = tflops / (acc.int8_tops * _EFF_HI) * 1e3   # best efficiency -> lowest latency
        ms_hi = tflops / (acc.int8_tops * _EFF_LO) * 1e3
        util_hi = ms_hi / (1e3 / _TARGET_FPS) * 100        # worst-case fraction of the frame budget
        return round(ms_lo, 2), round(ms_hi, 2), round(1e3 / ms_hi, 1), round(1e3 / ms_lo, 1), round(util_hi, 1)

    @staticmethod
    def _project_pair(det: Detector, acc: Accelerator, rows: dict, bank_cost: dict) -> ProjRow:
        gflops, work_mib = Projection._compute_cost(det, rows, bank_cost)
        envelope = acc.sram_mib
        util = round(work_mib / envelope * 100, 1) if envelope is not None else None
        ms_lo, ms_hi, fps_lo, fps_hi, cu_hi = Projection._latency(gflops, acc)
        return ProjRow(
            detector=det.name, accelerator=acc.name, gflops=round(gflops, 2), work_mem_mib=round(work_mib, 2),
            envelope_mib=envelope, mem_util_pct=util, onchip_fit=Projection._onchip_fit(work_mib, acc),
            latency_ms_lo=ms_lo, latency_ms_hi=ms_hi, fps_lo=fps_lo, fps_hi=fps_hi, compute_util_pct_hi=cu_hi,
            verdict=Projection._verdict(det, acc))

    @staticmethod
    def _rows(cost: dict, bank_cost: dict) -> list[ProjRow]:
        specs = accelerators.Accelerators.specs()
        return [Projection._project_pair(det, acc, cost, bank_cost)
                for det in Projection._DETECTORS for acc in specs]

    @staticmethod
    def _log(rows: list[ProjRow]) -> None:
        log.info(f"{'detector':11s} {'accelerator':16s} {'GFLOPs':>7s} {'mem(MiB)':>8s} {'onchip':>16s} "
                 f"{'lat(ms)band':>12s} {'FPS band':>13s} {'verdict':>18s}")
        for r in rows:
            fit = r.onchip_fit if r.mem_util_pct is None else f"{r.onchip_fit}({r.mem_util_pct:.0f}%)"
            lat = f"{r.latency_ms_lo}-{r.latency_ms_hi}" if r.latency_ms_lo is not None else "?"
            fps = f"{r.fps_lo}-{r.fps_hi}" if r.fps_lo is not None else "?"
            log.info(f"{r.detector:11s} {r.accelerator:16s} {r.gflops:7.1f} {r.work_mem_mib:8.1f} {fit:>16s} "
                     f"{lat:>12s} {fps:>13s} {r.verdict:>18s}")

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="structured projection output path")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        cost = Projection._load_cost(profile.DEFAULT_JSON)
        bank_cost = json.loads(bank.DEFAULT_JSON.read_text(encoding="utf-8"))
        rows = Projection._rows(cost, bank_cost)
        Projection._log(rows)
        payload = {"efficiency_band": [_EFF_LO, _EFF_HI], "target_fps": _TARGET_FPS,
                   "note": "projection (FLOPs/peak-TOPS x efficiency), not measured FPS",
                   "pairs": [asdict(r) for r in rows]}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log.info(f"wrote {args.json.relative_to(ROOT)}  ({len(rows)} pairs)")


SPEC = Spec("project", Projection.add_args, Projection.run)
