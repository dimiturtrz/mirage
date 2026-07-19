"""USD staging for the twin render entrypoints: turn parsed twin (verts, faces) into a UsdGeom.Mesh.
The pxr USD runtime only resolves once the kit app is up, so the import is local to build_mesh — the
module itself imports cleanly (numpy only), like control.sim.isaac_reach. The pure OBJ parse + geometry
helpers live in TwinGeom (pxr-free).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from pxr import Usd, UsdGeom


class TwinObj:
    @staticmethod
    def attr_float(attr: Usd.Attribute) -> float:
        # the pxr .pyi is Boost.Python-generated with every method typed `-> None`; Get returns the authored value
        return attr.Get()  # pyrefly: ignore[bad-return]

    @staticmethod
    def build_mesh(
        stage: Usd.Stage, verts: Float[np.ndarray, "v 3"], faces: Int[np.ndarray, "f 3"]
    ) -> UsdGeom.Mesh:
        from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

        mesh: UsdGeom.Mesh = UsdGeom.Mesh.Define(stage, "/World/Object")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten().astype(np.int32)))
        mesh.CreateDoubleSidedAttr(True)  # noqa: FBT003
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.7, 0.5)])
        return mesh
