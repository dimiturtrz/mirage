"""Prescriptive fit engine — cross each detector against each typed accelerator and say, with a reason,
whether it deploys and how well.

This is the deliverable the deploy work builds toward: not a FLOP table but a VERDICT. For every
(detector x accelerator x options) it resolves, in order, the gates that actually decide edge fit:

  1. op-fit      the detector's OP-CLASS vs the accelerator type's op-support — a conv graph is native
                 everywhere, a bank kNN tail is a host residue on commodity NPUs, attention is unsupported
                 on any fixed-function part. This is the load-bearing axis; FLOPs never enter it.
  2. quant gate  a full-int8 substrate (Coral/Hailo) will not load an fp32 graph at all.
  3. export gate a fixed-function NPU needs a compiled graph — with ONNX/TFLite export off there is no
                 eager fallback, so it cannot deploy; a CPU/GPU runs eager.
  4. memory-fit  the working set (weights at the chosen precision + peak activation + the bank in its
                 edge PQ form) against the instance's envelope UNDER its memory model — over an on-die-only
                 part (Hailo) it simply does not fit; over a scratchpad/unified part it streams at a penalty.
  5. latency     FLOPs / (peak-TOPS x a 30-70% efficiency band) where a TOPS number is cited — a PROJECTION,
                 never a measured FPS, and omitted where the datasheet number is behind a portal.

Emits deploy/fit_matrix.json — the (detector x accelerator x options) grid the viewer renders.

    python -m surfscan.deploy fit [--size 256]
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from enum import StrEnum

from core.obs import Obs
from surfscan.deploy import FIT_DOC, MODELS_DOC, accelerators
from surfscan.deploy.accelerators import Accelerator
from surfscan.deploy.schema import AccelType, MemoryModel, OpClass, OpSupport
from surfscan.dispatch import Spec

log = Obs.get()

_MIB = 1024 ** 2
_MB_PER_MIB = 1e6 / _MIB
_EFF_LO, _EFF_HI = 0.30, 0.70   # sustained fraction of peak TOPS (dataflow NPUs rarely exceed ~70% [roofline])
_TARGET_FPS = 30.0              # industrial inline budget the compute-utilization % is stated against
_INT8, _FP32 = "int8", "fp32"


@dataclass(frozen=True)
class Options:
    """The deploy toggles the fit adapts to — precision and whether an export toolchain is present."""
    quant: str   # _INT8 | _FP32
    onnx: bool   # a compiled ONNX/TFLite graph is available (required by fixed-function NPUs)


class Verdict(StrEnum):
    """The headline deployability verdict for one (detector x accelerator x options) cell."""
    FULL_ONDEVICE = "full-ondevice"     # whole graph on-accelerator, working set fits
    STREAMS = "streams-offchip"         # runs, but the working set is streamed from host (latency penalty)
    HOST_TAIL = "backbone+host-tail"    # conv on-accelerator, the kNN argmin/top-k tail on host CPU
    BLOCKED = "does-not-deploy"         # a gate fails: unsupported op / fp32 on int8-only / no export / over on-die


@dataclass(frozen=True)
class FitCell:
    """One (detector x accelerator x options) fit result — a verdict with a prescriptive reason."""
    detector: str
    op_class: OpClass
    accelerator: str
    accel_type: str
    quant: str
    onnx: bool
    op_support: OpSupport
    work_mib: float
    capacity_mib: float | None
    mem_util_pct: float | None
    gflops: float
    latency_ms_lo: float | None
    latency_ms_hi: float | None
    fps_lo: float | None
    fps_hi: float | None
    verdict: Verdict
    reason: str


class Fit:
    """Cross detectors x typed accelerators x options into a prescriptive per-cell deployability matrix."""

    @staticmethod
    def _work_mib(det: dict, comp: dict[str, dict], quant: str) -> tuple[float, float]:
        """(summed GFLOPs, working-set MiB) over a detector's pieces — conv stages AND banks alike, since a
        bank is a normal component (its disk is the store, its gflops the distance matmul). `pieces` is empty
        only for a detector with no on-device compute; a pure geometry-bank detector (BTF) lists its bank."""
        disk = "disk_int8_mb" if quant == _INT8 else "disk_fp32_mb"
        pieces = det["pieces"]
        gflops = sum(comp[p]["gflops"] for p in pieces)
        weights_mib = sum(comp[p][disk] * _MB_PER_MIB for p in pieces)
        act_mib = max((comp[p]["act_mb"] * _MB_PER_MIB for p in pieces), default=0.0)
        return round(gflops, 2), round(weights_mib + act_mib, 2)

    @staticmethod
    def _latency(gflops: float, acc: Accelerator):
        """Projected latency + FPS band from FLOPs / (peak-TOPS x efficiency). None where TOPS unsourced."""
        if acc.int8_tops is None:
            return None, None, None, None
        tflops = gflops / 1e3
        ms_lo = tflops / (acc.int8_tops * _EFF_HI) * 1e3
        ms_hi = tflops / (acc.int8_tops * _EFF_LO) * 1e3
        return round(ms_lo, 2), round(ms_hi, 2), round(1e3 / ms_hi, 1), round(1e3 / ms_lo, 1)

    @staticmethod
    def _blocking_reason(det: dict, acc: Accelerator, opt: Options, support: OpSupport) -> str | None:
        """The first gate that rejects the cell outright, as a prescriptive sentence — or None if it deploys."""
        if support == OpSupport.UNSUPPORTED:
            return f"{det['op_class']} is not in the {acc.type} op-set — runs only on CPU/GPU"
        if acc.int8_mandatory and opt.quant == _FP32:
            return f"{acc.name} requires full-int8 quantization; an fp32 graph will not load"
        if not opt.onnx and acc.type in {AccelType.NPU_FIXED, AccelType.NPU_SOC}:
            return f"{acc.type} needs a compiled graph — with export off there is no eager fallback"
        return None

    @staticmethod
    def _memory_verdict(work_mib: float, acc: Accelerator) -> tuple[Verdict, str]:
        """Working set vs the instance envelope under its memory model — fits, streams, or over on-die."""
        cap = acc.capacity_mib
        if cap is None or work_mib <= cap:
            return Verdict.FULL_ONDEVICE, "graph runs fully on-device — working set within the envelope"
        if acc.memory_model == MemoryModel.ON_DIE_ONLY:
            return Verdict.BLOCKED, (f"working set {work_mib:.1f} MiB exceeds the {cap:.0f} MiB on-die "
                                     f"envelope and there is no off-chip fallback — does not fit")
        return Verdict.STREAMS, (f"working set {work_mib:.1f} MiB over the {cap:.0f} MiB envelope — "
                                 f"streamed from host at a latency penalty")

    @staticmethod
    def _resolve(det: dict, acc: Accelerator, work_mib: float, opt: Options) -> tuple[Verdict, str]:
        """Run the gates in order; the first failure decides, else op-support + memory decide together."""
        support = acc.op_support[OpClass(det["op_class"])]
        blocked = Fit._blocking_reason(det, acc, opt, support)
        if blocked is not None:
            return Verdict.BLOCKED, blocked
        if support == OpSupport.HOST_TAIL:
            head = (f"conv backbone runs on {acc.name}, but the" if det["pieces"]
                    else "no accelerable neural graph — the FPFH descriptors and the")
            return Verdict.HOST_TAIL, (f"{head} kNN argmin/top-k tail is a host residue "
                                       f"({acc.type} has no on-device top-k over the bank)")
        return Fit._memory_verdict(work_mib, acc)

    @staticmethod
    def _cell(det: dict, acc: Accelerator, comp: dict, opt: Options) -> FitCell:
        op_class = OpClass(det["op_class"])
        gflops, work_mib = Fit._work_mib(det, comp, opt.quant)
        verdict, reason = Fit._resolve(det, acc, work_mib, opt)
        ms_lo, ms_hi, fps_lo, fps_hi = Fit._latency(gflops, acc)
        util = round(work_mib / acc.capacity_mib * 100, 1) if acc.capacity_mib is not None else None
        return FitCell(
            detector=det["name"], op_class=op_class, accelerator=acc.name,
            accel_type=acc.type, quant=opt.quant, onnx=opt.onnx, op_support=acc.op_support[op_class],
            work_mib=work_mib, capacity_mib=acc.capacity_mib, mem_util_pct=util, gflops=gflops,
            latency_ms_lo=ms_lo, latency_ms_hi=ms_hi, fps_lo=fps_lo, fps_hi=fps_hi,
            verdict=verdict, reason=reason)

    @staticmethod
    def matrix(models: dict) -> list[FitCell]:
        comp = {c["name"]: c for c in models["components"]}
        specs = accelerators.Accelerators.specs()
        opts = tuple(Options(quant, onnx) for quant, onnx in itertools.product((_INT8, _FP32), (True, False)))
        return [Fit._cell(det, acc, comp, opt)
                for det in models["detectors"] for acc in specs for opt in opts]

    @staticmethod
    def _log(cells: list[FitCell]) -> None:
        log.info(f"{'detector':11s} {'accelerator':16s} {'quant':5s} {'onnx':4s} {'work(MiB)':>9s} "
                 f"{'verdict':>18s}  reason")
        for c in cells:
            if not c.onnx:
                continue
            log.info(f"{c.detector:11s} {c.accelerator:16s} {c.quant:5s} {c.onnx!s:4s} {c.work_mib:9.1f} "
                     f"{c.verdict:>18s}  {c.reason}")

    @staticmethod
    def add_args(_ap: argparse.ArgumentParser) -> None:
        pass   # fit is a pure join over the committed models doc — nothing to measure

    @staticmethod
    def run(_args: argparse.Namespace) -> None:
        """Cross the committed model footprints with the typed accelerators, write the fit matrix."""
        models = Fit._models()
        cells = Fit.matrix(models)
        Fit._log(cells)
        doc = {
            "note": "projection (FLOPs/peak-TOPS x efficiency band), not measured FPS; op- and memory-fit are exact",
            "efficiency_band": [_EFF_LO, _EFF_HI], "target_fps": _TARGET_FPS,
            "input_size": models["input_size"], "device": models["device"],
            "cells": [asdict(c) for c in cells],
        }
        FIT_DOC.parent.mkdir(parents=True, exist_ok=True)
        FIT_DOC.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        log.info(f"wrote {FIT_DOC.name}  ({len(cells)} cells)")

    @staticmethod
    def _models() -> dict:
        """The committed model footprint (deploy/models_params.json). `fit` is a pure CPU join — it does not
        import the profiler (torch), so it stays runnable in CI without the heavy features extra."""
        if not MODELS_DOC.exists():
            raise FileNotFoundError(f"{MODELS_DOC.name} missing — run `python -m surfscan.deploy profile` first")
        return json.loads(MODELS_DOC.read_text(encoding="utf-8"))


SPEC = Spec("fit", Fit.add_args, Fit.run)
