"""Deploy vocabulary — the string values are a JSON contract with deployment/*.json, so pin them.

These enums round-trip through the hand-authored accelerator files and the generated model/fit docs; a
renamed value silently breaks the loader, so an equivalence-class check over each axis guards the wire form.
"""
from surfscan.deploy.schema import AccelType, MemoryModel, OpClass, OpSupport


class TestWireValues:
    def test_op_class_values(self):
        assert {c.value for c in OpClass} == {"conv_native", "bank_lookup", "transformer_attention"}

    def test_op_support_values(self):
        assert {s.value for s in OpSupport} == {"native", "host_tail", "unsupported"}

    def test_memory_model_values(self):
        assert {m.value for m in MemoryModel} == {"on_die_only", "scratchpad_host", "unified", "system"}

    def test_accel_type_values(self):
        assert {t.value for t in AccelType} == {"cpu", "gpu", "npu_fixed", "npu_soc"}


class TestStrEnumBehaviour:
    def test_constructs_from_wire_string(self):
        assert OpClass("bank_lookup") is OpClass.BANK_LOOKUP
        assert MemoryModel("on_die_only") is MemoryModel.ON_DIE_ONLY

    def test_is_str_for_json(self):
        assert OpSupport.HOST_TAIL == "host_tail"   # StrEnum serializes as its value
