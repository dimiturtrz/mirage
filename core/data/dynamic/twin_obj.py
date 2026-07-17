"""USD staging for the twin render entrypoints: turn parsed twin (verts, faces) into a UsdGeom.Mesh.
The pxr USD runtime only resolves once the kit app is up, so the import is local to build_mesh — the
module itself imports cleanly (numpy only), like control.sim.isaac_reach. The pure OBJ parse + geometry
helpers live in TwinGeom (pxr-free).
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int


class TwinObj:
    @staticmethod
    def build_mesh(stage, verts: Float[np.ndarray, "v 3"], faces: Int[np.ndarray, "f 3"]):
        from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

        mesh = UsdGeom.Mesh.Define(stage, "/World/Object")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten().astype(np.int32)))
        mesh.CreateDoubleSidedAttr(True)  # noqa: FBT003
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.7, 0.5)])
        return mesh
