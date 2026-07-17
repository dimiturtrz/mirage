"""Replicator one-frame smoke — p3h criterion #2: render a static scene and pull the label graph
(rgb + depth + normals + semantic seg). Proves the data-gen path end-to-end on the 5090.

Decisive doc-canonical run (Isaac Sim 6.0 headless static SDG) with every documented fix applied:
  - single GPU (IsaacSim #507 multi-GPU empty-frame): CUDA_VISIBLE_DEVICES=0 + multi_gpu False.
  - throttling ext async-toggle OFF (re-enables /app/asyncRendering when timeline stops -> frame skip).
  - explicit RealTimePathTracing renderer; standard render_product (not TiledCamera #4951).
  - TIMELINE PLAYING (RTX Sensor Annotators only collect while the timeline runs — 6.0 sensors doc).
  - step(delta_time=0.0, rt_subframes=32); 2 warmup steps (Windows first-frame skip).
  - DUAL readout: DiskBackend BasicWriter (6.0 explicit-backend API) AND raw annotators.get_data().
    both empty despite this => render pipeline not producing offscreen on this driver/install
    (escalate to driver 581.42 / full workstation install), not a script bug.

Run from sim/:  OMNI_KIT_ACCEPT_EULA=YES uv run python smoke_replicator.py
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> import core.obs
from isaacsim import SimulationApp

from core.obs import Obs

log = Obs.get()

OUT = os.path.join(os.path.dirname(__file__), "_smoke_out")
WDIR = os.path.join(OUT, "writer")
SUBFRAMES = 32

app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing",
                     "multi_gpu": False, "active_gpu": 0})
log.info("BOOTED")

import carb
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import torch
from pxr import Sdf, UsdLux

s = carb.settings.get_settings()
s.set_bool("/exts/isaacsim.core.throttling/enable_async", False)
s.set_bool("/app/asyncRendering", False)
s.set_int("/omni/replicator/RTSubframes", SUBFRAMES)

omni.usd.get_context().new_stage()
rep.orchestrator.set_capture_on_play(False)

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

backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir=WDIR)
writer = rep.writers.get("BasicWriter")
writer.initialize(backend=backend, rgb=True, distance_to_camera=True, normals=True,
                  semantic_segmentation=True)
writer.attach([rp])

timeline = omni.timeline.get_timeline_interface()
timeline.play()

log.info("RENDERING...")
for _ in range(2):                                       # warmup (Windows first-frame skip)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=SUBFRAMES)
for _ in range(3):                                       # capture
    rep.orchestrator.step(delta_time=0.0, rt_subframes=SUBFRAMES)
rep.orchestrator.wait_until_complete()
timeline.stop()

os.makedirs(OUT, exist_ok=True)
ann_ok = False
for n, a in annots.items():
    d = a.get_data()
    t = torch.as_tensor(d["data"] if isinstance(d, dict) else d)
    nz = bool(t.numel()) and bool(t.any())
    ann_ok = ann_ok or nz
    torch.save(t, os.path.join(OUT, f"{n}.pt"))
    log.info("  annot %-22s shape=%s nonzero=%s", n, tuple(t.shape), nz)

wfiles = os.listdir(WDIR) if os.path.isdir(WDIR) else []
log.info("  writer files=%d -> %s", len(wfiles), WDIR)
for f in sorted(wfiles)[:10]:
    log.info("    %s", f)

log.info("RESULT annot=%s writer=%s", 'PASS' if ann_ok else 'EMPTY', 'PASS' if wfiles else 'EMPTY')
app.close()
log.info("DONE")
