"""Compile-verified op-support — equivalence classes over the vendor-log parsers + the support classifier.

Pure text/logic: no vendor SDK is needed (the parsers consume compiler-log strings). One representative per
class — RKNN table split + NPU/CPU tag, EdgeTPU mapped-vs-fallback tag, and the NATIVE / HOST_TAIL /
host_fragmented verdict boundaries.
"""
from surfscan.deploy.accelerators import OpSupport
from surfscan.deploy.compile_probe import CompileProbe

_RKNN = """
D RKNN: ID   OpType   DataType Target InputShape OutputShape
D RKNN: 0    InputOperator  FLOAT16 CPU  \\  (1,16,64,64)
D RKNN: 1    ConvRelu       FLOAT16 NPU  (1,16,64,64)  (1,16,64,64)
D RKNN: 2    MaxPool        FLOAT16 NPU  (1,16,64,64)  (1,16,32,32)
D RKNN: 3    OutputOperator FLOAT16 CPU  (1,16,32,32)  \\
some unrelated line breaks the table
D RKNN: ID   OpType   DataType Target InputShape OutputShape
D RKNN: 1    Conv     FLOAT16 NPU  (1,64,64,64)  (1,256,64,64)
D RKNN: 2    TopK     FLOAT16 CPU  (1,256,64,64) (1,3,64,64)
D RKNN: 3    ArgMin   FLOAT16 CPU  (1,256,64,64) (1,1,64,64)
"""

_EDGETPU = """
==================== conv_rep ====================
CONV_2D        2   Mapped to Edge TPU
MAX_POOL_2D    1   Mapped to Edge TPU
==================== bank_tail ====================
CONV_2D        1   Mapped to Edge TPU
TOPK_V2        1   Operation not supported
ARG_MIN        1   Operation is working on an unsupported data type
"""


class TestParseRknn:
    def test_two_tables_split_and_named_in_build_order(self):
        blocks = CompileProbe.parse_rknn(_RKNN)
        assert list(blocks) == ["conv_rep", "bank_tail"]           # named by GRAPHS order, not by log text

    def test_io_operators_dropped_targets_tagged(self):
        conv = CompileProbe.parse_rknn(_RKNN)["conv_rep"]
        assert [o["op"] for o in conv] == ["ConvRelu", "MaxPool"]  # InputOperator/OutputOperator filtered
        assert {o["on"] for o in conv} == {"NPU"}


class TestParseEdgetpu:
    def test_mapped_vs_fallback_tagged(self):
        bank = CompileProbe.parse_edgetpu(_EDGETPU)["bank_tail"]
        by = {o["op"]: o["on"] for o in bank}
        assert by["CONV_2D"] == "EDGETPU"
        assert by["TOPK_V2"] == "CPU" and by["ARG_MIN"] == "CPU"


class TestVerdict:
    def test_all_native(self):
        assert CompileProbe._verdict([{"op": "Conv", "on": "NPU"}, {"op": "Relu", "on": "NPU"}]) == OpSupport.NATIVE

    def test_conv_plus_selection_tail_is_host_tail(self):
        rows = [{"op": "Conv", "on": "NPU"}, {"op": "TopK", "on": "CPU"}, {"op": "ArgMin", "on": "CPU"}]
        assert CompileProbe._verdict(rows) == OpSupport.HOST_TAIL

    def test_conv_on_host_is_fragmented_not_host_tail(self):
        rows = [{"op": "Conv", "on": "CPU"}, {"op": "Relu", "on": "CPU"}]
        assert CompileProbe._verdict(rows).startswith("host_fragmented")
