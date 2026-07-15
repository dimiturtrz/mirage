"""Candidate edge-accelerator spec table — the datasheet half of the deploy projection.

Structured, cited source for the accelerators a mirage detector could ship on. Every field traces to
research/deep_dives/2026-07-08_edge_accelerator_landscape.md (source tags [S#] there → primary vendor
docs). The load-bearing facts for the projection are (1) the on-chip memory envelope a bank/model must
fit, (2) the compute rate for a roofline latency band, and (3) whether the PatchCore kNN-vs-bank tail
(gather/argmin/top-k) runs on-device or is forced to host CPU — the constraint that splits "conv AE
fits everywhere" from "PatchCore's lookup is host-side on every commodity NPU".

Honesty boundary: numbers absent from the deep-dive are `None` (not guessed) — Coral/RKNN TOPS and the
Hailo on-die SRAM capacity sit behind vendor portals and were not fetched. The projection reports a
latency band only where a compute rate is known; the memory/op-fit verdict holds for every row.

    python -m surfscan.deploy accelerators     # emit docs/deployment/ACCELERATOR_SPECS.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import StrEnum

from core.obs import Obs
from surfscan.dispatch import Spec

log = Obs.get()

SRAM_CLASS_MIB = 8.0   # Coral EdgeTPU on-chip scratchpad; the "~8 MiB class" envelope the bank is sized against [S2][S5]
_SRC = "research/deep_dives/2026-07-08_edge_accelerator_landscape.md"


class Family(StrEnum):
    """Accelerator architecture class — the fixed-function NPUs vs the general GPU SoC."""
    DATAFLOW_NPU = "dataflow-npu"   # fixed-graph quantized-conv (Coral/Hailo/RKNN): no on-device kNN tail
    GPU_SOC = "gpu-soc"             # general GPU (Jetson): full ONNX, native top-k/argmin


@dataclass(frozen=True)
class Accelerator:
    """One candidate accelerator. `None` = spec not in the deep-dive (behind a portal / not fetched)."""
    name: str
    family: Family
    int8_tops: float | None     # peak int8 throughput (roofline numerator); None where unsourced
    fp_supported: bool          # non-int8 (fp16/bf16) compute available on-device
    int8_mandatory: bool        # full-int8 quantization required to run at all
    sram_mib: float | None      # on-chip memory envelope; None where capacity unquoted (Hailo DRAM-free)
    ext_memory: str             # off-chip memory story (host-streamed / DRAM-free on-die / GB LPDDR)
    gather: bool                # data-dependent gather on-device
    argmin: bool                # argmin/argmax over a dimension on-device
    topk: bool                  # top-k selection on-device
    patchcore_lookup_ondevice: bool  # can the kNN-vs-bank tail run on-accelerator (not host CPU)
    sources: str                # [S#] tags in the deep-dive
    note: str


class Accelerators:
    """The candidate edge-accelerator spec table — cited structured source for the deploy projection."""

    _SPECS = (
        Accelerator(
            name="coral_edgetpu", family=Family.DATAFLOW_NPU, int8_tops=None, fp_supported=False,
            int8_mandatory=True, sram_mib=SRAM_CLASS_MIB, ext_memory="host off-chip (streamed, latency penalty)",
            gather=False, argmin=False, topk=False, patchcore_lookup_ondevice=False, sources="S1,S2,S5",
            note="full-int8 mandatory; single-partition CPU fallback amputates the tail at first unsupported op"),
        Accelerator(
            name="hailo_8", family=Family.DATAFLOW_NPU, int8_tops=26.0, fp_supported=False,
            int8_mandatory=True, sram_mib=None, ext_memory="DRAM-free (all memory on-die)",
            gather=False, argmin=False, topk=False, patchcore_lookup_ondevice=False, sources="S6,S3,S7",
            note="build-time truncation at last CONV2D, pre/post on host; distance/top-k tail not expressible in dataflow"),
        Accelerator(
            name="hailo_8l", family=Family.DATAFLOW_NPU, int8_tops=13.0, fp_supported=False,
            int8_mandatory=True, sram_mib=None, ext_memory="DRAM-free (all memory on-die)",
            gather=False, argmin=False, topk=False, patchcore_lookup_ondevice=False, sources="S6,S3,S7",
            note="entry Hailo; same dataflow constraint as Hailo-8, half the TOPS"),
        Accelerator(
            name="rknn_npu", family=Family.DATAFLOW_NPU, int8_tops=None, fp_supported=True,
            int8_mandatory=False, sram_mib=None, ext_memory="external LPDDR (SoC-shared)",
            gather=True, argmin=True, topk=False, patchcore_lookup_ondevice=False, sources="S4",
            note="Gather/ArgMin on-device but TopK unsupported; k=1 argmin closer to feasible, fp32 bank still not native"),
        Accelerator(
            name="jetson_tensorrt", family=Family.GPU_SOC, int8_tops=None, fp_supported=True,
            int8_mandatory=False, sram_mib=None, ext_memory="GB-scale LPDDR shared CPU/GPU",
            gather=True, argmin=True, topk=True, patchcore_lookup_ondevice=True, sources="S8",
            note="general GPU: full ONNX, mixed precision, native TopK/ArgMin; runs PatchCore lookup end-to-end on-device"),
    )

    @staticmethod
    def specs() -> tuple[Accelerator, ...]:
        return Accelerators._SPECS

    @staticmethod
    def _log_table(specs: tuple[Accelerator, ...]) -> None:
        log.info(f"{'accelerator':16s} {'family':13s} {'int8_TOPS':>9s} {'SRAM(MiB)':>9s} "
                 f"{'gather':>6s} {'argmin':>6s} {'topk':>4s} {'lookup@dev':>10s}  ext_memory")
        for a in specs:
            tops = f"{a.int8_tops:.0f}" if a.int8_tops is not None else "?"
            sram = f"{a.sram_mib:.0f}" if a.sram_mib is not None else "?"
            log.info(f"{a.name:16s} {a.family:13s} {tops:>9s} {sram:>9s} {a.gather!s:>6s} "
                     f"{a.argmin!s:>6s} {a.topk!s:>4s} {a.patchcore_lookup_ondevice!s:>10s}  {a.ext_memory}")

    @staticmethod
    def section() -> dict:
        """The accelerator sub-document the projection embeds: cited specs + the sizing envelope."""
        return {"source": _SRC, "sram_class_mib": SRAM_CLASS_MIB,
                "accelerators": [asdict(a) for a in Accelerators.specs()]}

    @staticmethod
    def add_args(_ap: argparse.ArgumentParser) -> None:
        pass

    @staticmethod
    def run(_args: argparse.Namespace) -> None:
        Accelerators._log_table(Accelerators.specs())


SPEC = Spec("accelerators", Accelerators.add_args, Accelerators.run)
