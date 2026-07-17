"""USD staging for the sim/ Isaac render scripts: turn parsed twin (verts, faces) into a UsdGeom.Mesh.
Imported POST-SimulationApp-boot (the pxr USD runtime only resolves once the kit app is up), alongside
each script's other omni/pxr imports. The pure OBJ parse + geometry helpers live in twin_geom (pxr-free).
"""
import numpy as np
from pxr import Gf, UsdGeom, Vt


def build_mesh(stage, verts, faces):
    mesh = UsdGeom.Mesh.Define(stage, "/World/Object")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten().astype(np.int32)))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.7, 0.5)])
    return mesh
