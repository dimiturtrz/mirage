"""Shared vocabulary for the deployment fit model — the string enums both the JSON source files and the
Python fit engine speak, so a value like "host_tail" has one definition, not a literal sprinkled around.

Three axes decide deployability:
  * OpClass       — what kind of compute graph a detector IS (the load-bearing axis: conv ships anywhere,
                    a bank kNN tail or attention block does not).
  * OpSupport     — how an accelerator TYPE runs a given op-class: fully on-device, on-device with a host
                    residue, or not at all.
  * MemoryModel   — where an accelerator INSTANCE's working set lives (varies within a type: a fixed NPU
                    can be on-die-only like Hailo or scratchpad+host like Coral).
"""
from __future__ import annotations

from enum import StrEnum


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
