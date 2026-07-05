"""Replicator one-frame smoke — p3h criterion #2: render a scene and pull the label graph
(rgb + depth + normals + semantic seg) off a standard render product as torch tensors. Proves the
data-gen path end-to-end on the 5090.

Isaac Sim 6.0 canonical pattern + the headless-Windows SDG workarounds (per 6.0 replicator
troubleshooting): rep.functional.create.* (immediate prims), new_stage, set_capture_on_play(False),
disable async rendering (it skips frames), PLAY the timeline + warm up (Windows first-frame skip),
then orchestrator.step() to capture. Standard render_product (not the TiledCamera #4951 hang).

Run from sim/:  OMNI_KIT_ACCEPT_EULA=YES uv run python smoke_replicator.py
"""
import os

from isaacsim import SimulationApp

OUT = os.path.join(os.path.dirname(__file__), "_smoke_out")

app = SimulationApp({"headless": True})
print("BOOTED", flush=True)

import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Sdf, UsdLux  # noqa: E402

# Async rendering skips frames in headless SDG -> annotators come back empty. Disable it.
carb.settings.get_settings().set_bool("/app/asyncRendering", False)

omni.usd.get_context().new_stage()
rep.orchestrator.set_capture_on_play(False)

# Immediate prim creation (functional API).
rep.functional.create.xform(name="World")
rep.functional.create.cube(position=(0, 0, 0.5), parent="/World", semantics=[("class", "object")])
rep.functional.create.plane(scale=10, parent="/World", semantics=[("class", "ground")])
camera = rep.functional.create.camera(position=(4, 4, 3), look_at=(0, 0, 0), parent="/World")
light = UsdLux.DistantLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/Light"))
light.CreateIntensityAttr(3000.0)

rp = rep.create.render_product(camera, (512, 512))
names = ["rgb", "distance_to_camera", "normals", "semantic_segmentation"]
annots = {n: rep.annotators.get(n) for n in names}
for a in annots.values():
    a.attach(rp)

print("RENDERING...", flush=True)
for _ in range(10):            # warmup: materials/shaders load (async off; step() drives the frames)
    app.update()
for _ in range(3):
    rep.orchestrator.step()

os.makedirs(OUT, exist_ok=True)
for n, a in annots.items():
    d = a.get_data()
    t = torch.as_tensor(d["data"] if isinstance(d, dict) else d)   # Isaac returns numpy -> torch
    torch.save(t, os.path.join(OUT, f"{n}.pt"))
    print(f"  {n:22s} shape={tuple(t.shape)} dtype={t.dtype} nonzero={bool(t.any())}", flush=True)

print("WROTE ->", OUT, flush=True)
app.close()
print("DONE", flush=True)
