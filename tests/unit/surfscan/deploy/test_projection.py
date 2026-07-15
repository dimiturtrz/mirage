"""Deploy projection — equivalence classes over the two verdict axes and the latency band.

The op-verdict (conv-AE fits everywhere / PatchCore lookup is host-side except on Jetson) and the
on-chip-fit verdict (fits / streams-offchip / over-ondie / unknown) are the load-bearing outputs; the
latency band is present only where a compute rate is sourced.
"""
from surfscan.deploy.accelerators import Accelerator, Family
from surfscan.deploy.projection import Detector, OnchipFit, Projection, Verdict


def _acc(name, *, tops=None, sram=None, ext="host off-chip", lookup=False):
    return Accelerator(name=name, family=Family.DATAFLOW_NPU, int8_tops=tops, fp_supported=False,
                       int8_mandatory=True, sram_mib=sram, ext_memory=ext, gather=False, argmin=False,
                       topk=lookup, patchcore_lookup_ondevice=lookup, sources="Sx", note="")


_CONV = Detector("conv", ("m",), has_bank=False)
_BANK = Detector("pc", ("m",), has_bank=True)


class TestOpVerdict:
    def test_conv_ae_fits_ondevice_anywhere(self):
        assert Projection._verdict(_CONV, _acc("x")) == Verdict.FITS_ONDEVICE

    def test_patchcore_on_npu_is_backbone_plus_host_tail(self):
        assert Projection._verdict(_BANK, _acc("npu", lookup=False)) == Verdict.BACKBONE_ONLY

    def test_patchcore_on_lookup_capable_is_full_ondevice(self):
        assert Projection._verdict(_BANK, _acc("jetson", lookup=True)) == Verdict.FULL_ONDEVICE


class TestOnchipFit:
    def test_fits_when_under_envelope(self):
        assert Projection._onchip_fit(4.0, _acc("x", sram=8.0)) == OnchipFit.FITS

    def test_streams_when_over_and_offchip_exists(self):
        assert Projection._onchip_fit(40.0, _acc("coral", sram=8.0, ext="host off-chip")) == OnchipFit.STREAMS_OFFCHIP

    def test_over_ondie_when_dram_free(self):
        got = Projection._onchip_fit(40.0, _acc("hailo", sram=8.0, ext="DRAM-free (all memory on-die)"))
        assert got == OnchipFit.OVER_ONDIE

    def test_unknown_when_envelope_unsourced(self):
        assert Projection._onchip_fit(40.0, _acc("x", sram=None)) == OnchipFit.UNKNOWN


class TestLatencyBand:
    def test_none_without_tops(self):
        assert Projection._latency(10.0, _acc("x", tops=None)) == (None, None, None, None, None)

    def test_band_present_and_ordered_with_tops(self):
        ms_lo, ms_hi, fps_lo, fps_hi, cu_hi = Projection._latency(10.0, _acc("x", tops=26.0))
        assert 0 < ms_lo < ms_hi          # better efficiency -> lower latency
        assert 0 < fps_lo < fps_hi        # lower latency -> higher FPS
        assert cu_hi > 0
