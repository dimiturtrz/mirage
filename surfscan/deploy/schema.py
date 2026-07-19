"""Shared vocabulary for the deployment fit model — the string enums both the JSON source files and the
Python fit engine speak, so a value like "host_tail" has one definition, not a literal sprinkled around.

Three axes decide deployability:
  * OpClass       — what kind of compute graph a detector IS (the load-bearing axis: conv ships anywhere,
                    a bank kNN tail or attention block does not).
  * OpSupport     — how an accelerator TYPE runs a given op-class: fully on-device, on-device with a host
                    residue, or not at all.
  * MemoryModel   — where an accelerator INSTANCE's working set lives (varies within a type: a fixed NPU
                    can be on-die-only like Hailo or scratchpad+host like Coral).

The JSON wire contract the deploy steps read and write lives here too, one TypedDict per payload shape, so
each file's schema is stated once and both the writer and the reader are checked against it. An enum-valued
field is typed `str` wherever it is READ back from disk (json.loads yields a plain str) and `OpClass` only
where the value is produced in-process — an annotation must describe what is actually there.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NotRequired, TypedDict


@dataclass(frozen=True)
class CostRow:
    """One model's single-frame footprint — a neural forward OR a stored kNN bank, treated identically.

    A bank is a normal model here: its `params_m` is the stored vectors (N x C), its `gflops` the per-frame
    distance matmul, its `disk_fp32_mb` the raw store and `disk_int8_mb` the product-quantized edge form
    (a bank's quantization, as int8 weights are a conv model's). The argmin/top-k tail is not a footprint
    field — it rides the detector's op-class (BANK_LOOKUP), which the fit engine maps to a host residue."""
    name: str
    params_m: float
    gflops: float
    macs_g: float
    act_mb: float
    disk_fp32_mb: float
    disk_int8_mb: float
    note: str


class OpClass(StrEnum):
    """What kind of compute graph a detector is — the axis that decides on-device fit, not FLOP count."""
    CONV_NATIVE = "conv_native"                    # pure conv/deconv (ConvVAE, Draem, feat-recon)
    BANK_LOOKUP = "bank_lookup"                     # conv backbone + kNN distance/argmin/top-k tail (PatchCore)
    TRANSFORMER_ATTENTION = "transformer_attention"  # attention blocks (none of ours yet; the axis rejects them)


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


class ComponentCost(TypedDict):
    """`components[]` of deploy/models_params.json — a CostRow in its serialized form."""
    name: str
    params_m: float
    gflops: float
    macs_g: float
    act_mb: float
    disk_fp32_mb: float
    disk_int8_mb: float
    note: str


class DetectorDoc(TypedDict):
    """`detectors[]` of deploy/models_params.json — a composition of `components` names + its op-class."""
    name: str
    op_class: str          # an OpClass value; a plain str once read back from JSON
    pieces: list[str]


class ModelsDoc(TypedDict):
    """deploy/models_params.json — written by `profile`, read by `fit` and `export`."""
    note: str
    input_size: int
    device: str
    components: list[ComponentCost]
    detectors: list[DetectorDoc]


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


class ComponentOps(TypedDict):
    """`components[]` of deploy/export_ops.json. The export FAILING is a recorded result, so a row carries
    either the op inventory or the `error` — hence the NotRequired split, not a widened value type."""
    name: str
    exported: bool
    error: NotRequired[str]
    n_ops: NotRequired[int]
    op_types: NotRequired[list[str]]
    has_attention_op: NotRequired[bool]
    has_bank_tail_op: NotRequired[bool]


class DetectorOpsCheck(TypedDict):
    """`detectors[]` of deploy/export_ops.json — does the exported inventory corroborate the declared class?"""
    detector: str
    op_class: OpClass      # built in-process via OpClass(...), not read back
    pieces_exported: bool
    has_attention_op: bool
    corroborates_op_class: bool


class ExportDoc(TypedDict):
    """deploy/export_ops.json — the measured torch->ONNX op-class gate."""
    note: str
    opset: int
    input_size: int
    device: str
    components: list[ComponentOps]
    detectors: list[DetectorOpsCheck]


class GraphOps(TypedDict):
    """One representative graph emitted by `compile_probe --build`, with its exported ONNX op types."""
    graph: str
    represents: str        # an OpClass value
    ops: list[str]


class CompiledOp(TypedDict):
    """One row of a vendor compiler's op table. RKNN logs only (op, target); edgetpu_compiler also gives a
    count and the per-op status string — optional keys, not a widened row."""
    op: str
    on: str                # NPU | CPU | EDGETPU
    count: NotRequired[int]
    status: NotRequired[str]


class ToolchainDoc(TypedDict):
    """The vendor compiler a probe result came from — pinned so the measurement is reproducible."""
    tool: str
    version: str
    mode: str


class CompileResult(TypedDict):
    """`results[]` of deploy/compile_ops.json — one (representative graph x vendor target) partition."""
    graph: str
    represents: str        # an OpClass value
    target: str
    per_op: list[CompiledOp]
    verdict: str           # an OpSupport value, "no_data", or "host_fragmented(...)"


class ReconciliationDoc(TypedDict):
    """Per-op-class prose reconciling the compiled verdicts with what the fit matrix asserts."""
    conv_native: str
    bank_lookup: str
    transformer_attention: str


class CompileDoc(TypedDict):
    """deploy/compile_ops.json — compile-verified op-support, measured not cited."""
    note: str
    toolchains: dict[str, ToolchainDoc]
    gated: dict[str, str]
    results: list[CompileResult]
    reconciliation: ReconciliationDoc
