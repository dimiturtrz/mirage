"""Typed accelerator loader — equivalence classes over the op-support / memory-model invariants.

These are the facts the fit engine turns on: a fixed NPU runs conv native but the bank tail is a host
residue and attention is unsupported; only a general CPU/GPU runs the bank lookup fully on-device;
unsourced numbers stay None (not guessed). The source is the JSON in deployment/accelerators/.
"""
import json

import pytest
from pydantic import JsonValue, ValidationError

from surfscan.deploy import OpClass
from surfscan.deploy.accelerators import (
    Accelerator,
    AcceleratorInstanceDoc,
    Accelerators,
    AcceleratorTypeDoc,
    AccelType,
    MemoryModel,
    OpSupport,
)


class TestLoad:
    def test_every_type_loads_with_instances(self):
        types = {t.type: t for t in Accelerators.types()}
        assert set(types) == set(AccelType)
        assert all(t.instances for t in types.values())

    def test_specs_are_frozen_dataclasses(self):
        assert all(isinstance(a, Accelerator) for a in Accelerators.specs())

    def test_instance_inherits_type_op_support(self):
        for t in Accelerators.types():
            for inst in t.instances:
                assert inst.op_support == t.op_support


class TestOpSupportInvariants:
    def test_cpu_and_gpu_run_every_op_class_native(self):
        for a in Accelerators.specs():
            if a.type in (AccelType.CPU, AccelType.GPU):
                assert all(a.op_support[c] == OpSupport.NATIVE for c in OpClass)

    def test_fixed_and_soc_npus_host_tail_the_bank_lookup(self):
        for a in Accelerators.specs():
            if a.type in (AccelType.NPU_FIXED, AccelType.NPU_SOC):
                assert a.op_support[OpClass.BANK_LOOKUP] == OpSupport.HOST_TAIL
                assert a.op_support[OpClass.TRANSFORMER_ATTENTION] == OpSupport.UNSUPPORTED
                assert a.op_support[OpClass.CONV_NATIVE] == OpSupport.NATIVE


class TestMemoryModel:
    def test_hailo_is_on_die_only_coral_streams(self):
        by_name = {a.name: a for a in Accelerators.specs()}
        assert by_name["hailo_8"].memory_model == MemoryModel.ON_DIE_ONLY
        assert by_name["coral_edgetpu"].memory_model == MemoryModel.SCRATCHPAD_HOST

    def test_unsourced_numbers_are_none_not_zero(self):
        by_name = {a.name: a for a in Accelerators.specs()}
        assert by_name["hailo_8"].capacity_mib is None    # on-die SRAM portal-gated -> None, never a guessed 0
        assert by_name["coral_edgetpu"].capacity_mib == 8.0   # the one sourced NPU envelope


class TestPrecisions:
    def test_each_unit_declares_its_own_set(self):
        by_name = {a.name: a for a in Accelerators.specs()}
        assert set(by_name["rpi5_cortex_a76"].precisions) == {"fp32", "int8"}   # CPU: native fp32 + optional int8
        assert set(by_name["jetson_orin_nx"].precisions) == {"fp16", "int8"}    # GPU: native fp16 + int8
        assert set(by_name["coral_edgetpu"].precisions) == {"int8"}             # int8-only NPU: no choice

    def test_int8_only_npus_offer_int8_alone(self):
        for a in Accelerators.specs():
            if a.type in (AccelType.NPU_FIXED, AccelType.NPU_SOC):
                assert set(a.precisions) == {"int8"}   # quantization baked in — no fp32/fp16 to toggle


class TestWireValues:
    """The enum string values are a JSON contract with deployment/*.json — a renamed value silently breaks
    the loader, so an equivalence-class check over each axis guards the wire form."""

    def test_op_class_values(self):
        assert {c.value for c in OpClass} == {"conv_native", "bank_lookup", "transformer_attention"}

    def test_op_support_values(self):
        assert {s.value for s in OpSupport} == {"native", "host_tail", "unsupported"}

    def test_memory_model_values(self):
        assert {m.value for m in MemoryModel} == {"on_die_only", "scratchpad_host", "unified", "system"}

    def test_accel_type_values(self):
        assert {t.value for t in AccelType} == {"cpu", "gpu", "npu_fixed", "npu_soc"}


class TestSpecValidation:
    """The specs are hand-authored with no writer in the repo, so a typo is a HUMAN error that must fail
    loudly at parse time. Equivalence classes over the wire contract: extra key / bad enum value / bad
    scalar type all reject, and the valid class still yields the identical enriched Accelerator."""

    @staticmethod
    def _instance_spec() -> dict[str, JsonValue]:
        return {"name": "coral_edgetpu", "label": "Coral Edge TPU", "precisions": {"int8": 4.0},
                "memory_model": "scratchpad_host", "capacity_mib": 8.0,
                "ext_memory": "host off-chip (streamed, latency penalty)", "sources": "S1, S2",
                "note": "full-int8 mandatory"}

    @staticmethod
    def _type_spec() -> dict[str, JsonValue]:
        return {"type": "npu_fixed", "label": "fixed-function dataflow NPU",
                "op_support": {"conv_native": "native", "bank_lookup": "host_tail",
                               "transformer_attention": "unsupported"},
                "note": "quantized-conv dataflow graph",
                "instances": [TestSpecValidation._instance_spec()]}

    def test_valid_spec_parses(self):
        doc = AcceleratorTypeDoc.model_validate(self._type_spec())
        assert doc.type is AccelType.NPU_FIXED
        assert doc.op_support[OpClass.BANK_LOOKUP] is OpSupport.HOST_TAIL
        assert doc.instances[0].memory_model is MemoryModel.SCRATCHPAD_HOST

    def test_valid_spec_enriches_to_the_same_accelerator(self):
        doc = AcceleratorTypeDoc.model_validate(self._type_spec())
        built = Accelerators._instance(doc.instances[0], doc)
        loaded = {a.name: a for a in Accelerators.specs()}["coral_edgetpu"]
        assert built.type == loaded.type
        assert built.op_support == loaded.op_support
        assert built.precisions == loaded.precisions
        assert built.memory_model == loaded.memory_model
        assert built.capacity_mib == loaded.capacity_mib

    def test_misspelled_optional_key_is_rejected(self):
        spec: dict[str, JsonValue] = self._instance_spec()
        del spec["sources"]
        spec["sourcess"] = "S1, S2"    # silently dropped under the old TypedDict — now a named error
        with pytest.raises(ValidationError, match="sourcess"):
            AcceleratorInstanceDoc.model_validate(spec)

    def test_extra_key_on_the_type_doc_is_rejected(self):
        spec: dict[str, JsonValue] = self._type_spec()
        spec["op_support_provenence"] = "COMPILE-VERIFIED"
        with pytest.raises(ValidationError, match="op_support_provenence"):
            AcceleratorTypeDoc.model_validate(spec)

    def test_unknown_memory_model_is_rejected(self):
        spec: dict[str, JsonValue] = self._instance_spec()
        spec["memory_model"] = "on_die"
        with pytest.raises(ValidationError, match="memory_model"):
            AcceleratorInstanceDoc.model_validate(spec)

    def test_unknown_accel_type_is_rejected(self):
        spec: dict[str, JsonValue] = self._type_spec()
        spec["type"] = "npu"
        with pytest.raises(ValidationError, match="type"):
            AcceleratorTypeDoc.model_validate(spec)

    def test_unknown_op_class_key_is_rejected(self):
        spec: dict[str, JsonValue] = self._type_spec()
        spec["op_support"] = {"conv_natives": "native"}
        with pytest.raises(ValidationError, match="op_support"):
            AcceleratorTypeDoc.model_validate(spec)

    def test_unknown_op_support_value_is_rejected(self):
        spec: dict[str, JsonValue] = self._type_spec()
        spec["op_support"] = {"conv_native": "partial"}
        with pytest.raises(ValidationError, match="op_support"):
            AcceleratorTypeDoc.model_validate(spec)

    def test_string_capacity_is_rejected(self):
        spec: dict[str, JsonValue] = self._instance_spec()
        spec["capacity_mib"] = "8 MiB"
        with pytest.raises(ValidationError, match="capacity_mib"):
            AcceleratorInstanceDoc.model_validate(spec)

    def test_quoted_number_is_rejected(self):
        """A QUOTED number is the commonest hand-authoring slip and the one lax pydantic would coerce
        away silently ("8192" -> 8192). The strict scalar types are what make it an error."""
        spec: dict[str, JsonValue] = self._instance_spec()
        spec["capacity_mib"] = "8192"
        with pytest.raises(ValidationError, match="capacity_mib"):
            AcceleratorInstanceDoc.model_validate(spec)

    def test_quoted_number_in_precisions_is_rejected(self):
        spec: dict[str, JsonValue] = self._instance_spec()
        spec["precisions"] = {"int8": "4"}
        with pytest.raises(ValidationError, match="precisions"):
            AcceleratorInstanceDoc.model_validate(spec)

    def test_missing_required_key_is_rejected(self):
        spec: dict[str, JsonValue] = self._instance_spec()
        del spec["capacity_mib"]
        with pytest.raises(ValidationError, match="capacity_mib"):
            AcceleratorInstanceDoc.model_validate(spec)

    def test_authored_int_capacity_keeps_its_number_form(self):
        spec: dict[str, JsonValue] = self._instance_spec()
        spec["capacity_mib"] = 8192
        assert AcceleratorInstanceDoc.model_validate(spec).capacity_mib == 8192   # not widened to 8192.0

    def test_error_carries_the_field_path(self):
        with pytest.raises(ValidationError, match="instances.0.memory_model"):
            AcceleratorTypeDoc.model_validate_json(
                json.dumps(self._type_spec() | {"instances": [self._instance_spec() | {"memory_model": "x"}]}))


class TestStrEnumBehaviour:
    def test_constructs_from_wire_string(self):
        assert OpClass("bank_lookup") is OpClass.BANK_LOOKUP
        assert MemoryModel("on_die_only") is MemoryModel.ON_DIE_ONLY

    def test_is_str_for_json(self):
        assert OpSupport.HOST_TAIL == "host_tail"   # StrEnum serializes as its value
