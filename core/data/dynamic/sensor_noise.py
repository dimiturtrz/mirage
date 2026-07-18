"""jlc.2 — model the Zivid structured-light capture onto the clean path-traced twin xyz.

The twin renderer produces a geometrically-perfect surface; the real MVTec eval carries the Zivid sensor's
artifact statistics, and that render↔sensor domain gap is the twin's remaining deficit once resolution is
controlled (jlc.1: shape-source −0.224@256² → −0.067@512²). This injects the sensor character the twin lacks,
derived from real good-scan residuals (no magic numbers):

- **Axial jitter** along the local surface normal, σ(θ) = σ0 / cosθ — the structured-light triangulation
  error that grows with incidence angle θ (normal vs view). σ0 is the curvature-free noise floor measured on
  real good scans (2nd-difference, median ≈ 0.04 mm = 8e-4 store units).
- **Grazing dropout** — Zivid drops returns at grazing incidence / depth edges; real bbox holes average
  ~11 %. Modeled as p_drop(θ) = clip(k·(1/cosθ − 1), 0, pmax), k calibrated so the surface-mean hits the
  measured rate. The clean twin has a full mask; this punches the matching holes.
- **Flying-pixel spikes** — the heavy tail (real 2nd-diff p95 ≈ 51 mm): a small grazing-biased fraction of
  pixels gets a large axial displacement, the outlier cluster the preprocess SOR later fights (as on real).

A geometry-only model (xyz axis is the twin's contribution); rgb is passed through. Normals are recomputed
from the xyz map, so it runs post-hoc on the rendered/processed twin — no Isaac boot.
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from zlib import crc32

import numpy as np
from jaxtyping import Bool, Float

from core.data.static.store import Source, Store
from core.obs import Obs

log = Obs.get()
_EPS = 1e-9


@dataclass(frozen=True)
class SensorModel:
    """Zivid noise parameters, in the processed store's normalized units (SCALE = 0.05 m, so 1 unit = 50 mm).
    Defaults are the values derived from real MVTec good-scan residuals (jlc.2)."""
    sigma0: float = 8.0e-4        # axial floor σ (2nd-diff median, ≈ 0.04 mm)
    cos_min: float = 0.15         # clamp 1/cosθ so near-grazing normals don't blow up
    drop_mean: float = 0.11       # target surface-mean dropout (real bbox-hole rate)
    drop_max: float = 0.6         # per-pixel dropout ceiling at extreme grazing
    spike_frac: float = 0.02      # grazing-biased fraction of pixels that become flying pixels
    spike_scale: float = 0.02     # flying-pixel axial displacement scale (≈ 1 mm), heavy-tail multiplier


class ZividSensor:
    """Inject the derived Zivid structured-light artifacts into a clean twin xyz map."""

    def __init__(self, model: SensorModel | None = None):
        self._m = model or SensorModel()

    @staticmethod
    def _grazing(xyz: Float[np.ndarray, "h w 3"], valid: Bool[np.ndarray, "h w"]) -> Float[np.ndarray, "h w"]:
        """Incidence angle proxy cosθ = |n·z| from the normal of the smoothed xyz surface (image gradients)."""
        sm = xyz * valid[..., None]
        n = np.cross(np.gradient(sm, axis=1), np.gradient(sm, axis=0))
        nrm = np.linalg.norm(n, axis=-1)
        cos = np.where(nrm > _EPS, np.abs(n[..., 2]) / np.clip(nrm, _EPS, None), 1.0)
        return cos.astype(np.float32)

    def apply(self, xyz: Float[np.ndarray, "h w 3"], valid: Bool[np.ndarray, "h w"],
              rng: np.random.Generator) -> tuple[Float[np.ndarray, "h w 3"], Bool[np.ndarray, "h w"]]:
        """Clean twin (xyz, valid) → sensor-realistic (xyz', valid'): axial grazing jitter + flying-pixel
        spikes on the geometry, grazing-biased dropout on the mask. Invalid pixels stay zeroed."""
        m = self._m
        if not valid.any():
            return np.zeros_like(xyz), valid
        cos = np.clip(self._grazing(xyz, valid), m.cos_min, 1.0)
        inv_cos = 1.0 / cos                                            # 1/cosθ ≥ 1, the grazing amplifier

        z = np.array([0.0, 0.0, 1.0], np.float32)                     # axial ≈ view direction (top-down capture)
        jitter = rng.standard_normal(valid.shape).astype(np.float32) * (m.sigma0 * inv_cos)
        spike_hit = rng.random(valid.shape) < (m.spike_frac * (inv_cos - 1.0))
        spike = spike_hit * rng.standard_normal(valid.shape).astype(np.float32) * m.spike_scale
        xyz = xyz + (jitter + spike)[..., None] * z

        p_drop = np.clip(m.drop_max * (inv_cos - 1.0), 0.0, m.drop_max)
        p_drop = p_drop * (m.drop_mean / max(float(p_drop[valid].mean()), 1e-6))   # calibrate surface-mean
        keep = rng.random(valid.shape) >= np.clip(p_drop, 0.0, m.drop_max)
        valid = valid & keep

        xyz = (xyz * valid[..., None]).astype(np.float32)
        return xyz, valid

    @staticmethod
    def build_store(size: int, cats: list[str] | None = None) -> str:  # pragma: no cover
        """Transform the twin processed store → the twin_sensor store: apply the sensor model to each
        sample's xyz/valid (deterministic per sample_id), mirror the npz + meta.csv. A processed-space
        transform of `twin`, so no raw render / Isaac boot is needed."""
        sensor = ZividSensor()
        in_dir = Store.dataset_dir(size=size, source=Source.twin())
        out_dir = Store.dataset_dir(size=size, source=Source.twin_sensor())
        (out_dir / "data").mkdir(parents=True, exist_ok=True)
        df = Store.load(size=size, source=Source.twin())
        if cats:
            df = df.filter(df["category"].is_in(cats))
        for row in df.iter_rows(named=True):
            a = dict(np.load(in_dir / "data" / row["file"]))
            rng = np.random.default_rng(crc32(row["sample_id"].encode()))
            xyz, valid = sensor.apply(a["xyz"], a["valid"].astype(bool), rng)
            a["xyz"], a["valid"] = xyz, valid.astype(a["valid"].dtype)
            np.savez_compressed(out_dir / "data" / row["file"], **a)
        shutil.copy(in_dir / "meta.csv", out_dir / "meta.csv")
        log.info("twin_sensor store -> %s (%d samples, size=%d)", out_dir, len(df), size)
        return str(out_dir)

    @staticmethod
    def main() -> None:  # pragma: no cover  CLI
        ap = argparse.ArgumentParser(description="build the twin_sensor store (Zivid model on the twin, jlc.2)")
        ap.add_argument("--size", type=int, default=512)
        ap.add_argument("--cats", nargs="*", default=None)
        args = ap.parse_args()
        ZividSensor.build_store(args.size, args.cats)


if __name__ == "__main__":
    ZividSensor.main()
