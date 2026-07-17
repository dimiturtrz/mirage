"""Pure-numpy twin geometry helpers — no Isaac / USD dependency, so they import and unit-test without a
kit boot. OBJ parsing, z-depth back-projection to an organized xyz map, and the physical Gaussian-patch
defect. USD staging (build_mesh) + the render loop stay in the boot-only scripts / twin_obj.
"""
from __future__ import annotations

import numpy as np


def parse_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(verts, np.float32), np.array(faces, np.int32)


def backproject(depth, fx, fy):
    """z-depth (perpendicular) -> organized camera-frame xyz [H,W,3]; misses (inf) -> 0."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    x = (u - w / 2.0) / fx * z
    y = (v - h / 2.0) / fy * z
    return np.stack([x, y, z], -1).astype(np.float32)


def deform(verts, rng, sign):
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
