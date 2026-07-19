"""df0.2 step 1 — render a reconstructed twin mesh top-down and QC the shape.

Loads a df0.1 twin OBJ (D:/data/3d/twin/<cat>.obj) as a UsdGeom.Mesh in the original Zivid metric frame,
reproduces the top-down sensor pose (sensor at z~0 looking +z at the object at z~0.5), and renders rgb +
depth + normals via the headless Replicator path (timeline.play). Visual QC of the reconstruction before
the full good/defect randomization loop (generate_twin_synth).

The `isaacsim`/omni/pxr imports are local to the boot-driven methods, so the module imports without the
`sim` extra; only running boots the kit app. Run from the repo root:

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m core.data.dynamic.render_twin --cat bagel
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

from core.data.dynamic.twin_geom import TwinGeom
from core.data.dynamic.twin_obj import TwinObj
from core.obs import Obs

log = Obs.get()
_SUBFRAMES = 48
_RES = 512
_RENDER_STEPS = 3
_CLIP = (0.05, 100.0)
_CAM_X = 0.02                 # tiny x offset avoids a degenerate up-vector on the pure z-axis look
_DISTANT_LIGHT = 3000.0
_DOME_LIGHT = 1000.0


class TwinRenderQC:
    def __init__(self, cat: str):
        self._obj = f"D:/data/3d/twin/{cat}.obj"
        self._out = f"D:/data/3d/twin_qc/twin_{cat}"

    def run(self) -> None:
        import carb
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        verts, faces = TwinGeom.parse_obj(self._obj)
        centroid = verts.mean(0)
        log.info("mesh V=%d F=%d centroid=%s", len(verts), len(faces), centroid)

        s = carb.settings.get_settings()
        s.set_bool("/exts/isaacsim.core.throttling/enable_async", False)
        s.set_bool("/app/asyncRendering", False)
        s.set_int("/omni/replicator/RTSubframes", _SUBFRAMES)

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        rep.orchestrator.set_capture_on_play(False)
        TwinObj.build_mesh(stage, verts, faces)

        UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Light")).CreateIntensityAttr(_DISTANT_LIGHT)
        UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Dome")).CreateIntensityAttr(_DOME_LIGHT)

        cx, cy, cz = (float(v) for v in centroid)
        # reproduce the Zivid top-down pose: sensor near z=0 looking +z at the object (~0.52 m away)
        camera = rep.functional.create.camera(
            position=(cx + _CAM_X, cy, 0.0), look_at=(cx, cy, cz), parent="/World")
        UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera")).CreateClippingRangeAttr(Gf.Vec2f(*_CLIP))

        rp = rep.create.render_product(camera, (_RES, _RES))
        annots = {n: rep.annotators.get(n) for n in ("rgb", "distance_to_camera", "normals")}
        for a in annots.values():
            a.attach(rp)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        log.info("RENDERING...")
        for _ in range(_RENDER_STEPS):
            rep.orchestrator.step(delta_time=0.0, rt_subframes=_SUBFRAMES)
        rep.orchestrator.wait_until_complete()
        timeline.stop()

        self._save(annots)

    def _save(self, annots: dict[str, object]) -> None:
        os.makedirs(self._out, exist_ok=True)
        for n, a in annots.items():
            d = a.get_data()
            arr = d["data"] if isinstance(d, dict) else d
            t = torch.as_tensor(arr)
            torch.save(t, os.path.join(self._out, f"{n}.pt"))
            if n == "rgb":
                Image.fromarray(np.asarray(arr)[..., :3]).save(os.path.join(self._out, "rgb.png"))
            log.info("  %-20s shape=%s nonzero=%s", n, tuple(t.shape), bool(t.numel()) and bool(t.any()))
        log.info("WROTE -> %s", self._out)

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--cat", default="bagel")
        args = ap.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing",
                             "multi_gpu": False, "active_gpu": 0})
        log.info("BOOTED")
        TwinRenderQC(args.cat).run()
        app.close()
        log.info("DONE")


if __name__ == "__main__":
    TwinRenderQC.main()
