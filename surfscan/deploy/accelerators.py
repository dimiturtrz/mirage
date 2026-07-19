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
numbers behind a vendor portal are `null` in the JSON (not guessed), surfacing here as `None`. The JSON is
hand-authored with no writer in the codebase, so it is VALIDATED on read: the wire layer is a pair of
pydantic models that coerce every enum-valued field to its enum and forbid extra keys — a misspelled
optional key (`sourcess`) is a named parse error here, not a silently dropped value downstream.

    python -m surfscan.deploy accelerators     # log the loaded typed spec table
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt

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


class AcceleratorInstanceDoc(BaseModel):
    """`instances[]` of deploy/accelerators/<type>_params.json — one real unit's datasheet numbers,
    validated on read. `null` (-> None) is a spec behind a vendor portal, never a guess; `capacity_mib`
    keeps the authored JSON number form (an int MiB stays an int) so the emitted fit matrix is stable."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    label: str
    precisions: dict[str, StrictFloat | StrictInt]   # precision key -> peak TOPS
    memory_model: MemoryModel
    # Strict: lax pydantic coerces a QUOTED number ("8192") to 8192, which is the commonest slip in a
    # hand-authored JSON and exactly what this layer exists to catch. The int/float union (rather than
    # plain float) keeps the authored number form, so an int MiB is not widened to 8192.0 on re-emit.
    capacity_mib: StrictInt | StrictFloat | None
    ext_memory: str
    note: str
    sources: str = ""


class AcceleratorTypeDoc(BaseModel):
    """deploy/accelerators/<type>_params.json — a typed group sharing one op-support contract."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: AccelType
    label: str
    op_support: dict[OpClass, OpSupport]
    note: str
    instances: list[AcceleratorInstanceDoc]
    op_support_provenance: str = ""   # absent on cpu/gpu (nothing to compile-verify)
    source: str = ""                  # absent on cpu (cited per-instance instead)


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
    def _instance(inst: AcceleratorInstanceDoc, doc: AcceleratorTypeDoc) -> Accelerator:
        """Enrich one validated wire instance with the two axes it inherits from its type file."""
        return Accelerator(
            name=inst.name, label=inst.label, type=doc.type, op_support=doc.op_support,
            precisions=inst.precisions, memory_model=inst.memory_model,
            capacity_mib=inst.capacity_mib, ext_memory=inst.ext_memory,
            sources=inst.sources, note=inst.note)

    @staticmethod
    def _load_type(atype: AccelType) -> AcceleratorType:
        doc = AcceleratorTypeDoc.model_validate_json(
            (ACCEL_DIR / f"{atype}_params.json").read_text(encoding="utf-8"))
        instances = tuple(Accelerators._instance(i, doc) for i in doc.instances)
        return AcceleratorType(type=doc.type, label=doc.label, op_support=doc.op_support,
                               note=doc.note, instances=instances)

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
