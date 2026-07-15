"""Typed accelerator loader — equivalence classes over the op-support / memory-model invariants.

These are the facts the fit engine turns on: a fixed NPU runs conv native but the bank tail is a host
residue and attention is unsupported; only a general CPU/GPU runs the bank lookup fully on-device;
unsourced numbers stay None (not guessed). The source is the JSON in deployment/accelerators/.
"""
from surfscan.deploy.accelerators import Accelerator, Accelerators
from surfscan.deploy.schema import AccelType, MemoryModel, OpClass, OpSupport


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
        coral = next(a for a in Accelerators.specs() if a.name == "coral_edgetpu")
        assert coral.int8_tops is None          # not in the deep-dive -> None, never a guessed 0
        assert coral.capacity_mib == 8.0        # the one sourced envelope
