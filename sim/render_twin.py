"""df0.2 step 1 — render a reconstructed twin mesh top-down and QC the shape.

Loads a df0.1 twin OBJ (D:/data/3d/twin/<cat>.obj) as a UsdGeom.Mesh in the original Zivid metric
frame, reproduces the top-down sensor pose (sensor at z~0 looking +z at the object at z~0.5), and
renders rgb + depth + normals via the working headless Replicator path (timeline.play). This is the
visual QC of the reconstruction before the full good/defect randomization loop.

Run from sim/:  OMNI_KIT_ACCEPT_EULA=YES uv run python render_twin.py --cat bagel
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> import core.obs
from isaacsim import SimulationApp

from core.obs import Obs

log = Obs.get()

CAT = argparse.ArgumentParser()
CAT.add_argument("--cat", default="bagel")
ARGS = CAT.parse_args()

OBJ = f"D:/data/3d/twin/{ARGS.cat}.obj"
OUT = os.path.join(os.path.dirname(__file__), "_smoke_out", f"twin_{ARGS.cat}")
SUBFRAMES = 48

app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing",
                     "multi_gpu": False, "active_gpu": 0})
log.info("BOOTED")

import carb
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import torch
from pxr import Gf, Sdf, UsdGeom, UsdLux
from twin_geom import parse_obj
from twin_obj import build_mesh

verts, faces = parse_obj(OBJ)
centroid = verts.mean(0)
log.info("mesh V=%d F=%d centroid=%s", len(verts), len(faces), centroid)

s = carb.settings.get_settings()
s.set_bool("/exts/isaacsim.core.throttling/enable_async", False)
s.set_bool("/app/asyncRendering", False)
s.set_int("/omni/replicator/RTSubframes", SUBFRAMES)

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
rep.orchestrator.set_capture_on_play(False)

build_mesh(stage, verts, faces)

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

log.info("RENDERING...")
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
    log.info("  %-20s shape=%s nonzero=%s", n, tuple(t.shape), bool(t.numel()) and bool(t.any()))

log.info("WROTE -> %s", OUT)
app.close()
log.info("DONE")
