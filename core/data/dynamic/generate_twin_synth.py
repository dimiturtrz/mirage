"""df0.2 — twin synthetic-data generator: render the reconstructed twin meshes into MVTec-format good
scans via Replicator, with per-view pose + lighting randomization.

One Isaac session loops every category. Per view: jitter the object pose (in-plane rotation + small tilt +
translation) and the light intensities, render, then export the same representation MVTec carries — an rgb
png plus an organized camera-frame xyz position map (tiff), back-projected from the z-depth pass through
the pinhole intrinsics (the twin reproduces the Zivid top-down capture, so the xyz frame matches). Output
lands out-of-repo under <data>/synth/twin/<cat>/{train/good, test/<defect>}.

The `isaacsim`/omni/pxr imports are local to the boot-driven methods, so the module imports without the
`sim` extra; only running boots the kit app. Run from the repo root:

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m core.data.dynamic.generate_twin_synth --views 8
"""
from __future__ import annotations

import argparse
import os
from typing import Callable

import numpy as np
import tifffile
from jaxtyping import Float
from PIL import Image

from core.data.dynamic.twin_geom import TwinGeom
from core.data.dynamic.twin_obj import TwinObj
from core.obs import Obs

log = Obs.get()
_TWIN_DIR = "D:/data/3d/twin"
_OUT_ROOT = "D:/data/3d/synth/twin"
_SUBFRAMES = 48
_RENDER_STEPS = 3
_TILT = 8.0                       # max in-plane tilt (deg) about x/y
_SPIN = 360.0                     # full in-plane spin about z (deg)
_TRANS = 0.01                     # max centre translation (m)
_LIGHT = (2000.0, 4000.0)         # distant-light intensity range
_DOME = (600.0, 1400.0)           # dome-light intensity range
_CLIP = (0.05, 100.0)
_CAM_X = 0.02                     # tiny x offset avoids a degenerate up-vector on the pure z-axis look
_GT_AMP_FRAC = 0.3                # depth-diff gt threshold as a fraction of the deform amplitude
_DEF_FRAC = 3                     # defect views per type = views // _DEF_FRAC
_DEFECTS = (("dent", 1.0), ("bump", -1.0))


class TwinSynth:
    def __init__(self, views: int, res: int, cats: str | None):
        self._views = views
        self._res = res
        self._cats = cats
        self._rng = np.random.RandomState(0)

    def _categories(self) -> list[str]:
        if self._cats:
            return self._cats.split(",")
        return sorted(f[:-4] for f in os.listdir(_TWIN_DIR) if f.endswith(".obj"))

    def run(self) -> None:
        import carb
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt

        rng = self._rng
        s = carb.settings.get_settings()
        s.set_bool("/exts/isaacsim.core.throttling/enable_async", False)
        s.set_bool("/app/asyncRendering", False)
        s.set_int("/omni/replicator/RTSubframes", _SUBFRAMES)
        timeline = omni.timeline.get_timeline_interface()

        def intrinsics(stage: "pxr.Usd.Stage", w: int, h: int):
            cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
            focal = cam.GetFocalLengthAttr().Get()
            return (w * focal / cam.GetHorizontalApertureAttr().Get(),
                    h * focal / cam.GetVerticalApertureAttr().Get())

        def jitter_pose(mesh: "pxr.UsdGeom.Mesh"):
            api = UsdGeom.XformCommonAPI(mesh.GetPrim())
            api.SetRotate(Gf.Vec3f(float(rng.uniform(-_TILT, _TILT)), float(rng.uniform(-_TILT, _TILT)),
                                   float(rng.uniform(0, _SPIN))))
            api.SetTranslate(Gf.Vec3d(*(float(rng.uniform(-_TRANS, _TRANS)) for _ in range(3))))

        def render(rgb_a: "omni.replicator.core.Annotator", dep_a: "omni.replicator.core.Annotator"):
            for _ in range(_RENDER_STEPS):
                rep.orchestrator.step(delta_time=0.0, rt_subframes=_SUBFRAMES)
            return np.asarray(rgb_a.get_data())[..., :3], np.asarray(dep_a.get_data())

        for cat in self._categories():
            verts, faces = TwinGeom.parse_obj(f"{_TWIN_DIR}/{cat}.obj")
            centroid = verts.mean(0)
            cz = float(centroid[2])
            vc = (verts - centroid).astype(np.float32)         # centred so pose jitter + deform stay local

            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            rep.orchestrator.set_capture_on_play(False)
            mesh = TwinObj.build_mesh(stage, vc, faces)
            light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Light"))
            dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Dome"))

            camera = rep.functional.create.camera(position=(_CAM_X, 0.0, -cz), look_at=(0, 0, 0), parent="/World")
            UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera")).CreateClippingRangeAttr(Gf.Vec2f(*_CLIP))
            rp = rep.create.render_product(camera, (self._res, self._res))
            rgb_a = rep.annotators.get("rgb"); rgb_a.attach(rp)
            dep_a = rep.annotators.get("distance_to_image_plane"); dep_a.attach(rp)
            fx, fy = intrinsics(stage, self._res, self._res)

            def set_points(v: np.ndarray, mesh: "pxr.UsdGeom.Mesh" = mesh):
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))

            def randomize(
                mesh: "pxr.UsdGeom.Mesh" = mesh,
                light: "pxr.UsdLux.DistantLight" = light,
                dome: "pxr.UsdLux.DomeLight" = dome,
            ):
                jitter_pose(mesh)
                light.CreateIntensityAttr(float(rng.uniform(*_LIGHT)))
                dome.CreateIntensityAttr(float(rng.uniform(*_DOME)))

            timeline.play()
            npx = self._good_views(cat, vc, set_points, randomize, render, rgb_a, dep_a, fx, fy)
            n_def = self._defect_views(cat, vc, set_points, randomize, render, rgb_a, dep_a, fx, fy)
            timeline.stop()
            log.info("CAT %-12s good=%d defect=%d mean_obj_px=%d", cat, self._views, 2 * n_def, npx // self._views)
        log.info("DONE")

    def _good_views(
        self,
        cat: str,
        vc: Float[np.ndarray, "v 3"],
        set_points: Callable[[np.ndarray], None],
        randomize: Callable[[], None],
        render: Callable[["omni.replicator.core.Annotator", "omni.replicator.core.Annotator"],
                         tuple[np.ndarray, np.ndarray]],
        rgb_a: "omni.replicator.core.Annotator",
        dep_a: "omni.replicator.core.Annotator",
        fx: float,
        fy: float,
    ) -> int:
        gdir = f"{_OUT_ROOT}/{cat}/train/good"
        for sub in ("rgb", "xyz"):
            os.makedirs(f"{gdir}/{sub}", exist_ok=True)
        npx = 0
        for i in range(self._views):
            set_points(vc); randomize()
            rgb, depth = render(rgb_a, dep_a)
            xyz = TwinGeom.backproject(depth, fx, fy)
            Image.fromarray(rgb.astype(np.uint8)).save(f"{gdir}/rgb/{i:03d}.png")
            tifffile.imwrite(f"{gdir}/xyz/{i:03d}.tiff", xyz)
            npx += int((np.abs(xyz).sum(-1) > 0).sum())
        return npx

    def _defect_views(
        self,
        cat: str,
        vc: Float[np.ndarray, "v 3"],
        set_points: Callable[[np.ndarray], None],
        randomize: Callable[[], None],
        render: Callable[["omni.replicator.core.Annotator", "omni.replicator.core.Annotator"],
                         tuple[np.ndarray, np.ndarray]],
        rgb_a: "omni.replicator.core.Annotator",
        dep_a: "omni.replicator.core.Annotator",
        fx: float,
        fy: float,
    ) -> int:
        n_def = max(1, self._views // _DEF_FRAC)
        for dtype, sign in _DEFECTS:
            ddir = f"{_OUT_ROOT}/{cat}/test/{dtype}"
            for sub in ("rgb", "xyz", "gt"):
                os.makedirs(f"{ddir}/{sub}", exist_ok=True)
            for i in range(n_def):
                set_points(vc); randomize()
                _, good_depth = render(rgb_a, dep_a)                 # baseline at this exact pose
                dverts, amp = TwinGeom.deform(vc, self._rng, sign)
                set_points(dverts)
                rgb, depth = render(rgb_a, dep_a)                    # defect at the same pose
                xyz = TwinGeom.backproject(depth, fx, fy)
                obj = np.isfinite(depth) & np.isfinite(good_depth)
                gt = (obj & (np.abs(depth - good_depth) > amp * _GT_AMP_FRAC)).astype(np.uint8) * 255
                Image.fromarray(rgb.astype(np.uint8)).save(f"{ddir}/rgb/{i:03d}.png")
                tifffile.imwrite(f"{ddir}/xyz/{i:03d}.tiff", xyz)
                Image.fromarray(gt).save(f"{ddir}/gt/{i:03d}.png")
        return n_def

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--views", type=int, default=8, help="good views per category")
        ap.add_argument("--cats", default=None, help="comma-separated subset (default: all twin OBJs)")
        ap.add_argument("--res", type=int, default=512)
        args = ap.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing",
                             "multi_gpu": False, "active_gpu": 0})
        log.info("BOOTED")
        TwinSynth(args.views, args.res, args.cats).run()
        app.close()


if __name__ == "__main__":
    TwinSynth.main()
