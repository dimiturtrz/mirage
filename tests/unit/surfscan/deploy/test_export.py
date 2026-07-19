"""Measured ONNX export gate — equivalence classes over op-inventory classification + op-class corroboration.

Needs the `export` extra (onnx); skipped otherwise. A conv graph must inventory as conv-only (no attention,
no bank tail); the corroboration logic must accept a conv detector with a conv-only export and reject a
transformer detector whose export shows no attention op.
"""
import pytest

pytest.importorskip("onnx")

import torch
import torch.nn as nn

from surfscan.deploy import OpClass
from surfscan.deploy.export import Exporter


class TestOpInventory:
    def test_conv_model_is_conv_only(self):
        row = Exporter.export_one("tiny", nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU()), torch.randn(1, 3, 16, 16))
        assert row["exported"]
        assert not row["has_attention_op"] and not row["has_bank_tail_op"]

    def test_export_failure_is_recorded_not_raised(self):
        class Bad(nn.Module):
            def forward(self, x):
                raise RuntimeError("nope")
        row = Exporter.export_one("bad", Bad(), torch.randn(1, 3))
        assert row["exported"] is False and "error" in row


class TestCorroboration:
    @staticmethod
    def _det(name, op_class, pieces):
        return {"name": name, "op_class": op_class, "pieces": pieces}

    def test_conv_detector_corroborated_by_conv_only_export(self):
        by_name = {"m": {"name": "m", "exported": True, "has_attention_op": False}}
        chk = Exporter._detector_check(self._det("c", OpClass.CONV_NATIVE, ["m"]), by_name)
        assert chk["corroborates_op_class"]

    def test_transformer_detector_needs_an_attention_op(self):
        no_attn = {"t": {"name": "t", "exported": True, "has_attention_op": False}}
        chk = Exporter._detector_check(self._det("x", OpClass.TRANSFORMER_ATTENTION, ["t"]), no_attn)
        assert not chk["corroborates_op_class"]         # transformer that exports without attention = mismatch

    def test_transformer_with_attention_is_corroborated(self):
        attn = {"t": {"name": "t", "exported": True, "has_attention_op": True}}
        chk = Exporter._detector_check(self._det("x", OpClass.TRANSFORMER_ATTENTION, ["t"]), attn)
        assert chk["corroborates_op_class"]
