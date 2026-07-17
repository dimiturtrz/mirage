"""Pure-numpy twin geometry — no Isaac / USD dependency, so it imports and unit-tests without a kit boot.
OBJ parsing, z-depth back-projection to an organized xyz map, and the physical Gaussian-patch defect.
USD staging (TwinObj) + the render loop stay in the boot-driven entrypoints (generate_twin_synth /
render_twin), which import this library after the kit app is up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int


class TwinGeom:
    @staticmethod
    def parse_obj(path) -> tuple[Float[np.ndarray, "v 3"], Int[np.ndarray, "f 3"]]:
        verts, faces = [], []
        with Path(path).open() as fh:
            for line in fh:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
        return np.array(verts, np.float32), np.array(faces, np.int32)

    @staticmethod
    def backproject(depth: Float[np.ndarray, "h w"], fx: float, fy: float) -> Float[np.ndarray, "h w 3"]:
        """z-depth (perpendicular) -> organized camera-frame xyz [H,W,3]; misses (inf) -> 0."""
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        x = (u - w / 2.0) / fx * z
        y = (v - h / 2.0) / fy * z
        return np.stack([x, y, z], -1).astype(np.float32)

    @staticmethod
    def deform(verts: Float[np.ndarray, "v 3"], rng, sign: float) -> tuple[Float[np.ndarray, "v 3"], float]:
        """Physical defect: Gaussian z-displacement of a local surface patch (dent sign=+1 pushes the
        surface away from the sensor, bump sign=-1 toward it) — the 3D-mesh analogue of the classical
        normal-displacement defect, but rendered through the path tracer."""
        extent = float(np.linalg.norm(verts.max(0) - verts.min(0)))
        radius = extent * rng.uniform(0.05, 0.12)
        amp = sign * extent * rng.uniform(0.03, 0.06)
        c = verts[rng.randint(len(verts))]
        w = np.exp(-((verts - c) ** 2).sum(1) / (2 * radius ** 2))
        out = verts.copy()
        out[:, 2] += amp * w
        return out, abs(amp)
