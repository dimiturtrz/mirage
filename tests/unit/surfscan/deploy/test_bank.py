"""Bank memory model — equivalence classes over the byte formulas and the SRAM-fit verdicts.

The load-bearing correctness is (1) the PQ formula reproducing the PatchCore-Lite literature anchor
(~334 KiB for K=10000 d=256 [S3]) and (2) the fit/no-fit verdict flipping at the ~8 MiB envelope.
"""
from surfscan.deploy.accelerators import SRAM_CLASS_MIB
from surfscan.deploy.bank import BankCost, BankMemory


class TestByteFormulas:
    def test_fp32_is_four_bytes_per_element(self):
        c = BankMemory.cost("x", 1000, 512)
        assert c.fp32_mib == round(1000 * 512 * 4 / 1024 ** 2, 2)
        assert c.int8_mib == round(1000 * 512 / 1024 ** 2, 2)

    def test_pq_reproduces_patchcore_lite_anchor(self):
        # arxiv 2603.20288 [S3]: K=10000, d=256, m=8/b=8 -> ~334 KB PQ, ~9.77 MB fp32
        c = BankMemory.cost("anchor", 10_000, 256)
        assert 330 <= c.pq_kib <= 340
        assert 9.5 <= c.fp32_mib <= 10.0

    def test_pq_smaller_than_int8_smaller_than_fp32(self):
        c = BankMemory.cost("x", 20_000, 1536)
        assert c.pq_kib / 1024 < c.int8_mib < c.fp32_mib


class TestSramFit:
    def test_small_bank_fits_all_representations(self):
        c = BankMemory.cost("small", 100, 256)
        assert c.fp32_fits_sram and c.int8_fits_sram and c.pq_fits_sram

    def test_our_bank_fp32_and_int8_exceed_envelope_pq_fits(self):
        c = BankMemory.cost("ours", 20_000, 1536)   # 117 MiB fp32, 29 MiB int8, ~1.7 MiB PQ
        assert not c.fp32_fits_sram and not c.int8_fits_sram
        assert c.pq_fits_sram

    def test_verdict_flips_at_envelope(self):
        envelope_elems = int(SRAM_CLASS_MIB * 1024 ** 2)      # fp32 == 1 byte/elem boundary for int8
        below = BankMemory.cost("below", envelope_elems - 1, 1)
        above = BankMemory.cost("above", envelope_elems + 1, 1)
        assert below.int8_fits_sram and not above.int8_fits_sram


class TestSearchCost:
    def test_distance_and_argmin_scale_with_bank(self):
        c = BankMemory.cost("x", 20_000, 1536)
        assert c.distance_macs_g > 0 and c.argmin_compares_m > 0   # the host-side residue is non-trivial

    def test_cost_returns_dataclass(self):
        assert isinstance(BankMemory.cost("x", 10, 10), BankCost)


class TestRoles:
    def test_rgb_and_geometry_banks_present(self):
        roles = BankMemory.roles()
        assert set(roles) == {"rgb", "geometry"}
        assert roles["geometry"].channels == 33          # FPFH descriptor dim, fixed by Open3D
        assert roles["rgb"].fp32_mib > roles["geometry"].fp32_mib   # rgb (1536-d) bank dwarfs geometry (33-d)

    def test_section_banks_keyed_by_role(self):
        sec = BankMemory.section()
        assert set(sec["banks"]) == {"rgb", "geometry"}
        assert "anchor" in sec        # the PatchCore-Lite cross-check, not a deployable role
