"""Measured ONNX export gate — export each detector's neural graph and inventory its REAL ops.

The fit engine assigns each detector an op-class (conv_native / bank_lookup / transformer_attention). That
assignment was declared; here it is VERIFIED against the actual exported ONNX graph:

  * a conv detector exports to Conv/ConvTranspose/BatchNorm/... with NO Softmax   -> conv_native confirmed
  * the PatchCore backbone exports to a pure conv graph; its argmin/top-k tail is ABSENT from the export
    (it lives in host post-processing, not the model)                              -> host-tail corroborated
  * the Point-MAE point-transformer either exports WITH attention ops (Softmax) or fails to export at all
    on the commodity toolchain                                                     -> transformer_attention confirmed

Vendor compilation (EdgeTPU / Hailo / RKNN) is a separate hardware/SDK-gated step (bead j5s.5); this uses
only the local torch->ONNX exporter (dynamo=False) + the onnx op inventory. Emits deploy/export_ops.json.

    python -m surfscan.deploy export [--size 256]   # needs the `export` extra (onnx)
"""
from __future__ import annotations

import argparse
import io
import json
from typing import NotRequired, TypedDict, cast

import torch
from jaxtyping import Float
from torch import Tensor, nn

from core.obs import Obs
from surfscan.deploy import DEPLOY_DIR, OpClass, profile
from surfscan.deploy.profile import DetectorDoc, Exportable, ModelsDoc, ProfileTarget
from surfscan.dispatch import Spec

log = Obs.get()

_OPSET = 17
_ATTENTION_OPS = frozenset({"Softmax"})   # attention normalization: present in a transformer, absent in a conv graph
_BANK_TAIL_OPS = frozenset({"TopK", "ArgMin", "ArgMax"})  # the kNN tail — should NOT appear in an exported conv graph
_EXPORT_DOC = DEPLOY_DIR / "export_ops.json"


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


class Exporter:
    """Export each detector's neural graph to ONNX and inventory its ops — the measured op-class gate."""

    @staticmethod
    def _op_types(mod: nn.Module, xin: Float[Tensor, "b *shape"]) -> list[str]:
        buf = io.BytesIO()
        with torch.no_grad():
            # pyrefly: ignore[bad-argument-type]  stub narrows `f`; dynamo=False writes to a file-like object
            torch.onnx.export(mod, xin, buf, opset_version=_OPSET, dynamo=False)
        import onnx  # noqa: PLC0415  (optional `export` extra — imported only when the step actually runs)
        graph = onnx.load_from_string(buf.getvalue())
        return sorted({n.op_type for n in graph.graph.node})

    @staticmethod
    def export_one(name: str, fn: ProfileTarget, x: Float[Tensor, "b *shape"]) -> ComponentOps:
        mod, xin = (cast(Exportable, fn).export_spec(x) if hasattr(fn, "export_spec")
                    else (cast(nn.Module, fn), x))
        try:
            ops = Exporter._op_types(mod, xin)
        except Exception as exc:  # noqa: BLE001  the export FAILING is a recorded result (export-hostile graph)
            return {"name": name, "exported": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        opset = set(ops)
        return {"name": name, "exported": True, "n_ops": len(ops), "op_types": ops,
                "has_attention_op": bool(opset & _ATTENTION_OPS),
                "has_bank_tail_op": bool(opset & _BANK_TAIL_OPS)}

    @staticmethod
    def _detector_check(det: DetectorDoc, by_name: dict[str, ComponentOps]) -> DetectorOpsCheck:
        """Does the exported op inventory of a detector's pieces corroborate its declared op-class?"""
        rows = [by_name[p] for p in det["pieces"] if p in by_name]
        exported = all(r.get("exported") for r in rows)
        has_attn = any(r.get("has_attention_op") for r in rows)
        op_class = OpClass(det["op_class"])
        expect_attn = op_class == OpClass.TRANSFORMER_ATTENTION
        corroborated = exported and (has_attn == expect_attn) if exported else not expect_attn
        return {"detector": det["name"], "op_class": op_class, "pieces_exported": exported,
                "has_attention_op": has_attn, "corroborates_op_class": corroborated}

    @staticmethod
    def document(size: int, dev: str) -> ExportDoc:
        rows = [Exporter.export_one(name, fn, x) for name, fn, x, _ in profile.Profiler.targets(size, dev)]
        by_name = {r["name"]: r for r in rows}
        models: ModelsDoc = json.loads((profile.MODELS_DOC).read_text(encoding="utf-8")) \
            if profile.MODELS_DOC.exists() else profile.Profiler.document(size, dev)
        checks = [Exporter._detector_check(d, by_name) for d in models["detectors"]]
        return {
            "note": "measured torch->ONNX (dynamo=False) op inventory — the op-class gate, verified not assumed",
            "opset": _OPSET, "input_size": size, "device": dev,
            "components": rows, "detectors": checks,
        }

    @staticmethod
    def _log(doc: ExportDoc) -> None:
        for r in doc["components"]:
            if r["exported"]:
                flags = [f for f, on in (("attention", r["has_attention_op"]),
                                         ("bank-tail", r["has_bank_tail_op"])) if on] or ["conv-only"]
                log.info(f"{r['name']:18s} exported  {r['n_ops']:3d} ops  [{', '.join(flags)}]")
            else:
                log.info(f"{r['name']:18s} EXPORT FAILED  {r['error']}")
        for c in doc["detectors"]:
            mark = "ok" if c["corroborates_op_class"] else "MISMATCH"
            log.info(f"  {c['detector']:11s} {c['op_class']:22s} corroborated={mark}")

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--size", type=int, default=256, help="square input side (H=W) to export at")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        dev = profile.Profiler.device()
        doc = Exporter.document(args.size, dev)
        Exporter._log(doc)
        _EXPORT_DOC.parent.mkdir(parents=True, exist_ok=True)
        _EXPORT_DOC.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        log.info(f"wrote {_EXPORT_DOC.name}  ({len(doc['components'])} components exported/attempted)")


SPEC = Spec("export", Exporter.add_args, Exporter.run)
