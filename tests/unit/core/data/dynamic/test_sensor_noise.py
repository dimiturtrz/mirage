"""Unit tests for the Zivid sensor model (jlc.2) — the pure `ZividSensor.apply` transform.

Equivalence classes: (a) dropout only removes valid pixels (mask shrinks, never grows) and hits the derived
rate on a flat surface; (b) jitter perturbs only along the axial (z) direction and only on kept pixels;
(c) a fully-invalid input stays invalid (no spurious geometry); (d) deterministic under a seeded rng.
"""
import numpy as np

from core.data.dynamic.sensor_noise import SensorModel, ZividSensor


class TestZividSensor:
    @staticmethod
    def _tilted_plane(h=64, w=64):
        """A gently tilted plane in store units — a valid, non-degenerate surface with defined normals."""
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        z = 0.3 * (xs / w) + 0.1 * (ys / h)
        xyz = np.stack([xs / w - 0.5, ys / h - 0.5, z], -1).astype(np.float32)
        valid = np.ones((h, w), bool)
        return xyz, valid

    def test_dropout_only_shrinks_mask(self):
        xyz, valid = self._tilted_plane()
        _, v2 = ZividSensor().apply(xyz, valid, np.random.default_rng(0))
        assert v2.sum() < valid.sum()          # holes were punched
        assert not (v2 & ~valid).any()         # never validates a previously-invalid pixel

    def test_dropout_rate_near_derived_mean(self):
        xyz, valid = self._tilted_plane(128, 128)
        _, v2 = ZividSensor().apply(xyz, valid, np.random.default_rng(1))
        drop = 1.0 - v2.sum() / valid.sum()
        assert abs(drop - SensorModel().drop_mean) < 0.05   # calibrated to the real bbox-hole rate

    def test_jitter_is_axial_and_masked(self):
        xyz, valid = self._tilted_plane()
        xyz2, v2 = ZividSensor().apply(xyz, valid, np.random.default_rng(2))
        d = xyz2 - xyz
        assert np.allclose(d[v2][:, :2], 0.0)               # only z moved on kept pixels
        assert np.allclose(xyz2[~v2], 0.0)                  # dropped pixels are zeroed

    def test_all_invalid_stays_invalid(self):
        xyz = np.zeros((32, 32, 3), np.float32)
        valid = np.zeros((32, 32), bool)
        xyz2, v2 = ZividSensor().apply(xyz, valid, np.random.default_rng(3))
        assert not v2.any()
        assert np.allclose(xyz2, 0.0)

    def test_deterministic_under_seed(self):
        xyz, valid = self._tilted_plane()
        a = ZividSensor().apply(xyz, valid, np.random.default_rng(7))
        b = ZividSensor().apply(xyz, valid, np.random.default_rng(7))
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
