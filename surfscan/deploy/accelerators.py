"""Typed accelerator spec loader — reads the hand-authored, cited source in deploy/accelerators/.

The source of truth is now the browsable JSON, not this module: deploy/accelerators/<type>_params.json,
one file per accelerator TYPE (cpu / gpu / npu_fixed / npu_soc), real instances inside. The type carries
the op-support contract (how it runs each op-class); the instance carries its datasheet numbers and its
memory model (which varies within a type — Coral streams from host, Hailo is on-die-only, both npu_fixed).

Both axes an accelerator declares about ITSELF are defined here, beside the loader that reads them:
`OpSupport` (how a TYPE runs a given op-class — fully on-device, on-device with a host residue, or not at
all) and `MemoryModel` (where an INSTANCE's working set lives — it varies within a type: a fixed NPU can
be on-die-only like Hailo or scratchpad+host like Coral). The fit engine consumes both; neither is its.

Every field traces to research/deep_dives/2026-07-08_edge_accelerator_landscape.md (source tags [S#]);
numbers behind a vendor portal are `null` in the JSON (not guessed), surfacing here as `None`. An
enum-valued field is typed `str` where it is READ back from disk (json.loads yields a plain str) and as
its enum only where the value is produced in-process — an annotation must describe what is actually there.

    python -m surfscan.deploy accelerators     # log the loaded typed spec table
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from typing import NotRequired, TypedDict

from core.obs import Obs
from surfscan.deploy import ACCEL_DIR, OpClass
from surfscan.dispatch import Spec

log = Obs.get()


class OpSupport(StrEnum):
    """How an accelerator type runs an op-class — the prescriptive verdict primitive."""
    NATIVE = "native"          # whole graph runs on-accelerator
    HOST_TAIL = "host_tail"    # conv part on-accelerator, argmin/top-k residue on host CPU
    UNSUPPORTED = "unsupported"  # op-class not expressible on this substrate at all


class MemoryModel(StrEnum):
    """Where an accelerator instance's working set lives — sets whether an over-envelope model still runs."""
    ON_DIE_ONLY = "on_die_only"        # Hailo: no external RAM; over the on-die envelope = does not fit
    SCRATCHPAD_HOST = "scratchpad_host"  # Coral: small on-chip scratchpad, streamed from host (latency penalty)
    UNIFIED = "unified"                # Jetson/RKNN: GB-scale shared LPDDR
    SYSTEM = "system"                  # CPU: system RAM


class AccelType(StrEnum):
    """Accelerator family — the typed file a set of instances is grouped under."""
    CPU = "cpu"
    GPU = "gpu"
    NPU_FIXED = "npu_fixed"    # fixed-function dataflow NPU (Coral, Hailo)
    NPU_SOC = "npu_soc"        # SoC-integrated NPU (RKNN)


class AcceleratorInstanceDoc(TypedDict):
    """`instances[]` of deploy/accelerators/<type>_params.json — one real unit's datasheet numbers.
    `null` (-> None) is a spec behind a vendor portal, never a guess."""
    name: str
    label: str
    precisions: dict[str, float]   # precision key -> peak TOPS
    memory_model: str              # a MemoryModel value
    capacity_mib: float | None
    ext_memory: str
    note: str
    sources: NotRequired[str]


class AcceleratorTypeDoc(TypedDict):
    """deploy/accelerators/<type>_params.json — a typed group sharing one op-support contract."""
    type: str                      # an AccelType value
    label: str
    op_support: dict[str, str]     # OpClass value -> OpSupport value
    note: str
    instances: list[AcceleratorInstanceDoc]
    op_support_provenance: NotRequired[str]   # absent on cpu/gpu (nothing to compile-verify)
    source: NotRequired[str]                  # absent on cpu (cited per-instance instead)


@dataclass(frozen=True)
class Accelerator:
    """One accelerator instance: datasheet numbers + its memory model, plus the type's op-support contract.
    `None` = spec not in the deep-dive (behind a portal / not fetched)."""
    name: str
    label: str
    type: AccelType
    op_support: dict[OpClass, OpSupport]  # inherited from the type — how each op-class runs here
    precisions: dict[str, float]          # the precisions THIS unit runs, precision -> peak TOPS (native first):
                                          # CPU {fp32,int8}, GPU {fp16,int8}, int8-only NPU {int8}
    memory_model: MemoryModel
    capacity_mib: float | None            # working-set envelope; None where unquoted
    ext_memory: str
    sources: str
    note: str


@dataclass(frozen=True)
class AcceleratorType:
    """A typed group of accelerators sharing an op-support contract (one deploy/accelerators file)."""
    type: AccelType
    label: str
    op_support: dict[OpClass, OpSupport]
    note: str
    instances: tuple[Accelerator, ...] = field(default_factory=tuple)


class Accelerators:
    """Load the typed, cited accelerator specs from deploy/accelerators/ — the source-of-truth JSON."""

    @staticmethod
    def _support_map(raw: dict[str, str]) -> dict[OpClass, OpSupport]:
        return {OpClass(k): OpSupport(v) for k, v in raw.items()}

    @staticmethod
    def _instance(
        raw: AcceleratorInstanceDoc, atype: AccelType, support: dict[OpClass, OpSupport]
    ) -> Accelerator:
        return Accelerator(
            name=raw["name"], label=raw["label"], type=atype, op_support=support,
            precisions=raw["precisions"], memory_model=MemoryModel(raw["memory_model"]),
            capacity_mib=raw["capacity_mib"], ext_memory=raw["ext_memory"],
            sources=raw.get("sources", ""), note=raw["note"])

    @staticmethod
    def _load_type(atype: AccelType) -> AcceleratorType:
        raw: AcceleratorTypeDoc = json.loads((ACCEL_DIR / f"{atype}_params.json").read_text(encoding="utf-8"))
        support = Accelerators._support_map(raw["op_support"])
        instances = tuple(Accelerators._instance(i, atype, support) for i in raw["instances"])
        return AcceleratorType(type=atype, label=raw["label"], op_support=support,
                               note=raw["note"], instances=instances)

    @staticmethod
    @cache
    def types() -> tuple[AcceleratorType, ...]:
        return tuple(Accelerators._load_type(t) for t in AccelType)

    @staticmethod
    def specs() -> tuple[Accelerator, ...]:
        return tuple(inst for t in Accelerators.types() for inst in t.instances)

    @staticmethod
    def _log_table(specs: tuple[Accelerator, ...]) -> None:
        log.info(f"{'accelerator':16s} {'type':10s} {'precisions (TOPS)':>22s} {'cap(MiB)':>9s} {'memory':>16s}  "
                 f"conv / bank / attn")
        for a in specs:
            prec = ", ".join(f"{p}:{t:g}" for p, t in a.precisions.items())
            cap = f"{a.capacity_mib:.0f}" if a.capacity_mib is not None else "?"
            ops = " / ".join(a.op_support[c] for c in OpClass)
            log.info(f"{a.name:16s} {a.type:10s} {prec:>22s} {cap:>9s} {a.memory_model:>16s}  {ops}")

    @staticmethod
    def add_args(_ap: argparse.ArgumentParser) -> None:
        pass

    @staticmethod
    def run(_args: argparse.Namespace) -> None:
        Accelerators._log_table(Accelerators.specs())


SPEC = Spec("accelerators", Accelerators.add_args, Accelerators.run)
