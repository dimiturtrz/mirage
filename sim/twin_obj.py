"""Shared twin-OBJ loader for the sim/ Isaac render scripts: parse a df0.1 twin OBJ into (verts, faces)
and stage the corresponding UsdGeom.Mesh. Imported POST-SimulationApp-boot (the pxr USD runtime only
resolves once the kit app is up), alongside each script's other omni/pxr imports.
"""
import numpy as np
from pxr import Gf, UsdGeom, Vt


def parse_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(verts, np.float32), np.array(faces, np.int32)


def build_mesh(stage, verts, faces):
    mesh = UsdGeom.Mesh.Define(stage, "/World/Object")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten().astype(np.int32)))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.7, 0.5)])
    return mesh
