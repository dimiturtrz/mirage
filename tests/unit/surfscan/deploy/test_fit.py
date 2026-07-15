"""Fit engine — equivalence classes over the gate pipeline that decides edge deployability.

One representative per gate outcome: op-fit (native / host-tail / unsupported), the quant gate (fp32 on
an int8-only part), the export gate (no compiled graph on an NPU), and memory-fit (fits / streams / over
an on-die-only envelope). The bank tail on a fixed NPU and the whole graph on a GPU are the thesis cells.
"""
from surfscan.deploy.accelerators import Accelerator, Accelerators
from surfscan.deploy.fit import Fit, Options, Verdict
from surfscan.deploy.schema import AccelType, MemoryModel, OpClass, OpSupport

_CONV = {"name": "c", "op_class": OpClass.CONV_NATIVE, "pieces": ["m"], "has_bank": False}
_BANK = {"name": "p", "op_class": OpClass.BANK_LOOKUP, "pieces": ["m"], "has_bank": True}
_COMP = {"m": {"name": "m", "gflops": 1.0, "disk_int8_mb": 1.0, "disk_fp32_mb": 4.0, "act_mb": 1.0}}
_BANKSEC = {"banks": [{"pq_kib": 1024.0, "fp32_mib": 100.0}],
            "host_search_per_frame": {"distance_macs_g": 1.0}}


class _Make:
    @staticmethod
    def models(detector: dict) -> dict:
        return {"input_size": 256, "device": "cpu", "components": list(_COMP.values()),
                "detectors": [detector], "bank": _BANKSEC}

    @staticmethod
    def accel(name: str, atype: AccelType, mem: MemoryModel, cap, int8_only: bool) -> Accelerator:
        types = {t.type: t for t in Accelerators.types()}
        return Accelerator(name=name, label=name, type=atype, op_support=types[atype].op_support,
                           int8_tops=None, fp_supported=not int8_only, int8_mandatory=int8_only,
                           memory_model=mem, capacity_mib=cap, ext_memory="", sources="", note="")


class TestOpFit:
    def test_conv_on_cpu_is_full_ondevice(self):
        cell = Fit._cell(_CONV, _real("rpi5_cortex_a76"), _COMP, _BANKSEC, Options("int8", onnx=True))
        assert cell.verdict == Verdict.FULL_ONDEVICE

    def test_bank_on_fixed_npu_is_host_tail(self):
        cell = Fit._cell(_BANK, _real("hailo_8"), _COMP, _BANKSEC, Options("int8", onnx=True))
        assert cell.verdict == Verdict.HOST_TAIL
        assert cell.op_support == OpSupport.HOST_TAIL

    def test_bank_on_gpu_is_full_ondevice(self):
        cell = Fit._cell(_BANK, _real("jetson_orin"), _COMP, _BANKSEC, Options("int8", onnx=True))
        assert cell.verdict == Verdict.FULL_ONDEVICE


class TestGates:
    def test_fp32_on_int8_only_part_is_blocked(self):
        cell = Fit._cell(_CONV, _real("hailo_8"), _COMP, _BANKSEC, Options("fp32", onnx=True))
        assert cell.verdict == Verdict.BLOCKED

    def test_no_export_on_npu_is_blocked(self):
        cell = Fit._cell(_CONV, _real("hailo_8"), _COMP, _BANKSEC, Options("int8", onnx=False))
        assert cell.verdict == Verdict.BLOCKED

    def test_no_export_on_cpu_still_runs(self):
        cell = Fit._cell(_CONV, _real("rpi5_cortex_a76"), _COMP, _BANKSEC, Options("int8", onnx=False))
        assert cell.verdict == Verdict.FULL_ONDEVICE


class TestMemoryFit:
    def test_over_scratchpad_streams(self):
        acc = _Make.accel("small_coral", AccelType.NPU_FIXED, MemoryModel.SCRATCHPAD_HOST, 1.0, int8_only=False)
        cell = Fit._cell(_CONV, acc, _COMP, _BANKSEC, Options("int8", onnx=True))
        assert cell.verdict == Verdict.STREAMS

    def test_over_on_die_only_does_not_fit(self):
        acc = _Make.accel("tiny_hailo", AccelType.NPU_FIXED, MemoryModel.ON_DIE_ONLY, 1.0, int8_only=False)
        cell = Fit._cell(_CONV, acc, _COMP, _BANKSEC, Options("int8", onnx=True))
        assert cell.verdict == Verdict.BLOCKED

    def test_within_envelope_fits(self):
        acc = _Make.accel("big", AccelType.NPU_FIXED, MemoryModel.ON_DIE_ONLY, 10_000.0, int8_only=False)
        cell = Fit._cell(_CONV, acc, _COMP, _BANKSEC, Options("int8", onnx=True))
        assert cell.verdict == Verdict.FULL_ONDEVICE


class TestMatrix:
    def test_matrix_covers_all_option_combos(self):
        cells = Fit.matrix(_Make.models(_CONV))
        n_acc = len(Accelerators.specs())
        assert len(cells) == n_acc * 4          # one detector x n accelerators x (int8/fp32 x onnx on/off)


def _real(name: str) -> Accelerator:
    return next(a for a in Accelerators.specs() if a.name == name)
