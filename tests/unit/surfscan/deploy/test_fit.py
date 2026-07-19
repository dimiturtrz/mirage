"""Fit engine — equivalence classes over the gate pipeline that decides edge deployability.

One representative per gate outcome: op-fit (native / host-tail / unsupported), the export gate (no compiled
graph on an NPU), memory-fit (fits / streams / over an on-die-only envelope), and precision (each accelerator
crosses only its OWN supported precisions). A bank is just another piece (a component), so a bank_lookup
detector lists its bank in `pieces`; the host-tail on a fixed NPU and the whole graph on a GPU are the thesis.
"""
from surfscan.deploy import OpClass
from surfscan.deploy.accelerators import Accelerator, Accelerators, AccelType, MemoryModel, OpSupport
from surfscan.deploy.fit import Fit, Options, Verdict

_CONV = {"name": "c", "op_class": OpClass.CONV_NATIVE, "pieces": ["m"]}
_BANK = {"name": "p", "op_class": OpClass.BANK_LOOKUP, "pieces": ["m", "bank1"]}   # a bank is just a piece
_COMP = {"m": {"name": "m", "gflops": 1.0, "disk_int8_mb": 1.0, "disk_fp32_mb": 4.0, "act_mb": 1.0},
         "bank1": {"name": "bank1", "gflops": 2.0, "disk_int8_mb": 1.7, "disk_fp32_mb": 100.0, "act_mb": 0.0}}
_INT8 = Options("int8", onnx=True)


class _Make:
    @staticmethod
    def models(detector: dict) -> dict:
        return {"input_size": 256, "device": "cpu", "components": list(_COMP.values()), "detectors": [detector]}

    @staticmethod
    def accel(name: str, atype: AccelType, mem: MemoryModel, cap, precisions=None) -> Accelerator:
        types = {t.type: t for t in Accelerators.types()}
        return Accelerator(name=name, label=name, type=atype, op_support=types[atype].op_support,
                           precisions=precisions or {"int8": 1.0}, memory_model=mem, capacity_mib=cap,
                           ext_memory="", sources="", note="")


class TestOpFit:
    def test_conv_on_cpu_is_full_ondevice(self):
        assert Fit._cell(_CONV, _real("rpi5_cortex_a76"), _COMP, _INT8).verdict == Verdict.FULL_ONDEVICE

    def test_bank_on_fixed_npu_is_host_tail(self):
        cell = Fit._cell(_BANK, _real("hailo_8"), _COMP, _INT8)
        assert cell.verdict == Verdict.HOST_TAIL and cell.op_support == OpSupport.HOST_TAIL

    def test_bank_on_gpu_is_full_ondevice(self):
        assert Fit._cell(_BANK, _real("jetson_orin_nx"), _COMP, _INT8).verdict == Verdict.FULL_ONDEVICE

    def test_bank_storage_counts_toward_working_set(self):
        fp16 = Options("fp16", onnx=True)
        assert Fit._cell(_BANK, _real("jetson_orin_nx"), _COMP, fp16).work_mib > \
               Fit._cell(_CONV, _real("jetson_orin_nx"), _COMP, fp16).work_mib


class TestGates:
    def test_no_export_on_npu_is_blocked(self):
        assert Fit._cell(_CONV, _real("hailo_8"), _COMP, Options("int8", onnx=False)).verdict == Verdict.BLOCKED

    def test_no_export_on_cpu_still_runs(self):
        assert Fit._cell(_CONV, _real("rpi5_cortex_a76"), _COMP, Options("int8", onnx=False)).verdict \
               == Verdict.FULL_ONDEVICE


class TestPrecisionPerAccelerator:
    def test_npu_matrix_has_only_int8_cells(self):
        cells = [c for c in Fit.matrix(_Make.models(_CONV)) if c.accelerator == "hailo_8"]
        assert {c.precision for c in cells} == {"int8"}          # int8-only NPU -> no fp32/fp16 cell exists

    def test_cpu_matrix_has_fp32_and_int8(self):
        cells = [c for c in Fit.matrix(_Make.models(_CONV)) if c.accelerator == "rpi5_cortex_a76"]
        assert {c.precision for c in cells} == {"fp32", "int8"}

    def test_fp16_weights_are_half_fp32(self):
        gpu = _real("jetson_orin_nx")
        fp16 = Fit._cell(_CONV, gpu, _COMP, Options("fp16", onnx=True)).work_mib
        fp32 = Fit._cell(_CONV, gpu, _COMP, Options("fp32", onnx=True)).work_mib   # fp32 disk still scales even if not offered
        assert abs(fp16 - fp32 / 2) < 0.6      # fp16 weights = half fp32 (act term makes it approximate)


class TestComputeThroughput:
    def test_cpu_gets_a_latency_band(self):
        cell = Fit._cell(_CONV, _real("rpi5_cortex_a76"), _COMP, _INT8)
        assert cell.latency_ms_lo is not None and cell.fps_hi > 0

    def test_int8_faster_than_float_on_gpu(self):
        i8 = Fit._cell(_CONV, _real("jetson_orin_nx"), _COMP, _INT8)
        fp16 = Fit._cell(_CONV, _real("jetson_orin_nx"), _COMP, Options("fp16", onnx=True))
        assert i8.latency_ms_lo < fp16.latency_ms_lo   # int8 rate (50) beats fp16 (25)

    def test_unsupported_precision_has_no_band(self):
        cell = Fit._cell(_CONV, _real("coral_edgetpu"), _COMP, Options("fp16", onnx=True))
        assert cell.latency_ms_lo is None   # coral has no fp16 rate -> no latency for that precision


class TestMemoryFit:
    def test_over_scratchpad_streams(self):
        acc = _Make.accel("small_coral", AccelType.NPU_FIXED, MemoryModel.SCRATCHPAD_HOST, 1.0)
        assert Fit._cell(_CONV, acc, _COMP, _INT8).verdict == Verdict.STREAMS

    def test_over_on_die_only_does_not_fit(self):
        acc = _Make.accel("tiny_hailo", AccelType.NPU_FIXED, MemoryModel.ON_DIE_ONLY, 1.0)
        assert Fit._cell(_CONV, acc, _COMP, _INT8).verdict == Verdict.BLOCKED

    def test_within_envelope_fits(self):
        acc = _Make.accel("big", AccelType.NPU_FIXED, MemoryModel.ON_DIE_ONLY, 10_000.0)
        assert Fit._cell(_CONV, acc, _COMP, _INT8).verdict == Verdict.FULL_ONDEVICE


class TestMatrix:
    def test_matrix_covers_each_units_precisions(self):
        cells = Fit.matrix(_Make.models(_CONV))
        expected = sum(len(a.precisions) * 2 for a in Accelerators.specs())   # precisions x (onnx on/off)
        assert len(cells) == expected


def _real(name: str) -> Accelerator:
    return next(a for a in Accelerators.specs() if a.name == name)
