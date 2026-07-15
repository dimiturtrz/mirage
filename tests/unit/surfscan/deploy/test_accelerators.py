"""Accelerator spec table — equivalence classes over the op-support / lookup-feasibility invariants.

These are the facts the deploy projection turns on: only Jetson runs the PatchCore lookup on-device;
every commodity NPU forces the argmin/top-k tail to host; unsourced numbers stay None (not guessed).
"""
from surfscan.deploy.accelerators import Accelerator, Accelerators


class TestSpecInvariants:
    def test_only_jetson_runs_lookup_ondevice(self):
        on = [a.name for a in Accelerators.specs() if a.patchcore_lookup_ondevice]
        assert on == ["jetson_tensorrt"]

    def test_lookup_ondevice_requires_topk(self):
        # the tail needs top-k; no commodity NPU has it, so none run the lookup on-device
        for a in Accelerators.specs():
            if a.patchcore_lookup_ondevice:
                assert a.topk

    def test_dataflow_npus_are_int8_or_have_no_topk(self):
        for a in Accelerators.specs():
            if a.family == "dataflow-npu":
                assert not a.patchcore_lookup_ondevice

    def test_every_spec_cites_a_source(self):
        for a in Accelerators.specs():
            assert a.sources and a.sources.startswith("S")

    def test_unsourced_numbers_are_none_not_zero(self):
        coral = next(a for a in Accelerators.specs() if a.name == "coral_edgetpu")
        assert coral.int8_tops is None          # not in the deep-dive -> None, never a guessed 0
        assert coral.sram_mib == 8.0            # the one sourced envelope

    def test_specs_are_frozen_dataclasses(self):
        assert all(isinstance(a, Accelerator) for a in Accelerators.specs())
