"""Bank cost model — equivalence classes over the byte formulas and the bank-as-normal-model shape.

The load-bearing correctness is (1) the PQ formula reproducing the PatchCore-Lite literature anchor
(~334 KB for K=10000 d=256 [S3]) and (2) a bank emitting as an ordinary CostRow — params = N x C, gflops
= the distance matmul, int8 disk = the product-quantized edge form — so the fit engine treats it like any
conv stage (no bank special-case).
"""
from surfscan.deploy.bank import GEOMETRY_BANK, RGB_BANK, BankMemory, CostRow

_PATCHES = 1024


class TestByteFormulas:
    def test_fp32_is_four_bytes_per_element(self):
        c = BankMemory.cost("x", 1000, 512, _PATCHES, "n")
        assert c.disk_fp32_mb == round(1000 * 512 * 4 / 1e6, 2)
        assert c.params_m == round(1000 * 512 / 1e6, 2)

    def test_pq_int8_reproduces_patchcore_lite_anchor(self):
        # arxiv 2603.20288 [S3]: K=10000, d=256, m=8/b=8 -> ~334 KB PQ (~0.34 MB), ~10 MB fp32
        c = BankMemory.cost("anchor", 10_000, 256, _PATCHES, "n")
        assert 0.33 <= c.disk_int8_mb <= 0.35
        assert 9.5 <= c.disk_fp32_mb <= 10.5

    def test_pq_edge_form_much_smaller_than_fp32(self):
        c = BankMemory.cost("x", 20_000, 1536, _PATCHES, "n")
        assert c.disk_int8_mb < c.disk_fp32_mb / 10   # PQ is the whole reason a bank fits edge


class TestBankAsModel:
    def test_cost_is_an_ordinary_costrow(self):
        assert isinstance(BankMemory.cost("x", 10, 10, _PATCHES, "n"), CostRow)

    def test_params_are_the_stored_vectors(self):
        c = BankMemory.cost("x", 20_000, 1536, _PATCHES, "n")
        assert c.params_m == round(20_000 * 1536 / 1e6, 2)

    def test_gflops_is_the_distance_matmul(self):
        c = BankMemory.cost("x", 20_000, 1536, _PATCHES, "n")
        assert c.gflops == round(2 * _PATCHES * 20_000 * 1536 / 1e9, 2) and c.gflops > 0

    def test_no_persistent_activation(self):
        assert BankMemory.cost("x", 20_000, 1536, _PATCHES, "n").act_mb == 0.0   # search is a streamed reduction


class TestComponents:
    def test_rgb_and_geometry_banks_present(self):
        rows = {r.name: r for r in BankMemory.components()}
        assert set(rows) == {RGB_BANK, GEOMETRY_BANK}
        assert rows[RGB_BANK].disk_fp32_mb > rows[GEOMETRY_BANK].disk_fp32_mb   # rgb (1536-d) dwarfs geometry (33-d)
        assert all(r.disk_int8_mb < r.disk_fp32_mb for r in rows.values())
