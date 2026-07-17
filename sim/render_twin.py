"""df0.2 step 1 — render a reconstructed twin mesh top-down and QC the shape.

Loads a df0.1 twin OBJ (D:/data/3d/twin/<cat>.obj) as a UsdGeom.Mesh in the original Zivid metric
frame, reproduces the top-down sensor pose (sensor at z~0 looking +z at the object at z~0.5), and
renders rgb + depth + normals via the working headless Replicator path (timeline.play). This is the
visual QC of the reconstruction before the full good/defect randomization loop.

Run from sim/:  OMNI_KIT_ACCEPT_EULA=YES uv run python render_twin.py --cat bagel
"""
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from isaacsim import SimulationApp

CAT = argparse.ArgumentParser()
CAT.add_argument("--cat", default="bagel")
ARGS = CAT.parse_args()

OBJ = f"D:/data/3d/twin/{ARGS.cat}.obj"
OUT = os.path.join(os.path.dirname(__file__), "_smoke_out", f"twin_{ARGS.cat}")
SUBFRAMES = 48

app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing",
                     "multi_gpu": False, "active_gpu": 0})
print("BOOTED", flush=True)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt  # noqa: E402


def parse_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(verts, np.float32), np.array(faces, np.int32)


verts, faces = parse_obj(OBJ)
centroid = verts.mean(0)
print(f"mesh V={len(verts)} F={len(faces)} centroid={centroid}", flush=True)

s = carb.settings.get_settings()
s.set_bool("/exts/isaacsim.core.throttling/enable_async", False)
s.set_bool("/app/asyncRendering", False)
s.set_int("/omni/replicator/RTSubframes", SUBFRAMES)

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
rep.orchestrator.set_capture_on_play(False)

mesh = UsdGeom.Mesh.Define(stage, "/World/Object")
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten().astype(np.int32)))
mesh.CreateDoubleSidedAttr(True)                       # single-sided surface -> don't backface-cull
mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
mesh.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.7, 0.5)])

light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Light"))
light.CreateIntensityAttr(3000.0)
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Dome"))
dome.CreateIntensityAttr(1000.0)

cx, cy, cz = (float(v) for v in centroid)
extent = float(np.linalg.norm(verts.max(0) - verts.min(0)))
# reproduce the Zivid top-down pose: sensor near z=0 looking +z at the object (~0.52 m away).
# tiny x offset avoids a degenerate up-vector on the pure z-axis look.
camera = rep.functional.create.camera(
    position=(cx + 0.02, cy, 0.0), look_at=(cx, cy, cz), parent="/World")
UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera")).CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))

rp = rep.create.render_product(camera, (512, 512))
names = ["rgb", "distance_to_camera", "normals"]
annots = {n: rep.annotators.get(n) for n in names}
for a in annots.values():
    a.attach(rp)

timeline = omni.timeline.get_timeline_interface()
timeline.play()

print("RENDERING...", flush=True)
for _ in range(3):
    rep.orchestrator.step(delta_time=0.0, rt_subframes=SUBFRAMES)
rep.orchestrator.wait_until_complete()
timeline.stop()

os.makedirs(OUT, exist_ok=True)
for n, a in annots.items():
    d = a.get_data()
    arr = d["data"] if isinstance(d, dict) else d
    t = torch.as_tensor(arr)
    torch.save(t, os.path.join(OUT, f"{n}.pt"))
    if n == "rgb":
        from PIL import Image
        Image.fromarray(np.asarray(arr)[..., :3]).save(os.path.join(OUT, "rgb.png"))
    print(f"  {n:20s} shape={tuple(t.shape)} nonzero={bool(t.numel()) and bool(t.any())}", flush=True)

print(f"WROTE -> {OUT}", flush=True)
app.close()
print("DONE", flush=True)
