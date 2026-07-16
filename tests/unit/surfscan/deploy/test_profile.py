"""Deploy profiler — equivalence-class units for the measured-footprint math (CPU, no network).

The footprint arithmetic (params->disk, FLOPs->MACs, cpu act=nan) and the backbone-truncation
selection (layer4/fc excluded) are the load-bearing bits; the torchvision download + cuda act peak are
the exec boundary, not unit-tested here.
"""
import pytest

pytest.importorskip("torchvision")   # the profiler builds a torchvision backbone; skip where the features extra is absent

import torch
import torch.nn as nn

from surfscan.deploy.profile import CostRow, Profiler, TruncatedResnet


class TestTruncatedResnetSelection:
    def test_layer2_excludes_layer3_and_beyond(self):
        names = TruncatedResnet.used_names("layer2")
        assert names[-1] == "layer2"
        assert "layer3" not in names and "layer4" not in names and "fc" not in names

    def test_layer3_includes_layer3_excludes_layer4_fc(self):
        names = TruncatedResnet.used_names("layer3")
        assert "layer3" in names
        assert "layer4" not in names and "fc" not in names and "avgpool" not in names

    def test_stem_always_present(self):
        assert TruncatedResnet.used_names("layer2")[:4] == ["conv1", "bn1", "relu", "maxpool"]


class TestProfileOne:
    @staticmethod
    def _tiny():
        return nn.Conv2d(32, 64, 3, padding=1).eval()   # ~18.5k params, non-zero after 2dp rounding

    @staticmethod
    def _x():
        return torch.randn(1, 32, 64, 64)

    def test_row_fields_positive_and_disk_ratio(self):
        model = self._tiny()
        params = Profiler._params(model)
        row = Profiler.profile_one("tiny", model, self._x(), "note", "cpu")
        assert isinstance(row, CostRow)
        assert row.params_m > 0 and row.gflops > 0
        assert row.disk_int8_mb == round(params / 1e6, 2)           # int8 = 1 byte/param
        assert row.disk_fp32_mb == round(params * 4 / 1e6, 2)       # fp32 = 4 bytes/param
        assert 0 < row.macs_g <= row.gflops                          # MACs = FLOPs / 2 (<= FLOPs)

    def test_cpu_activation_is_nan(self):
        row = Profiler.profile_one("tiny", self._tiny(), self._x(), "note", "cpu")
        assert row.act_mb != row.act_mb   # nan off cuda

    def test_params_counts_module_weights(self):
        conv = self._tiny()
        expected = sum(p.numel() for p in conv.parameters())
        assert Profiler._params(conv) == expected
