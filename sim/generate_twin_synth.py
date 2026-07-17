"""df0.2 — twin synthetic-data generator: render the reconstructed twin meshes into MVTec-format
good scans via Replicator, with per-view pose + lighting randomization.

One Isaac session loops every category. Per view: jitter the object pose (in-plane rotation + small
tilt + translation) and the light intensities, render, then export the same representation MVTec
carries — an rgb png plus an organized camera-frame xyz position map (tiff), back-projected from the
z-depth pass through the pinhole intrinsics (the twin reproduces the Zivid top-down capture, so the
xyz frame matches). Output lands out-of-repo under <data>/synth/twin/<cat>/train/good/.

Part 1 emits GOOD views only (also a generalization check that all 10 reconstructions render).
Defect views + gt masks follow in part 2.

Run from sim/:  OMNI_KIT_ACCEPT_EULA=YES uv run python generate_twin_synth.py --views 8
"""
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from isaacsim import SimulationApp

AP = argparse.ArgumentParser()
AP.add_argument("--views", type=int, default=8, help="good views per category")
AP.add_argument("--cats", default=None, help="comma-separated subset (default: all twin OBJs)")
AP.add_argument("--res", type=int, default=512)
ARGS = AP.parse_args()

TWIN_DIR = "D:/data/3d/twin"
OUT_ROOT = "D:/data/3d/synth/twin"
SUBFRAMES = 48

app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing",
                     "multi_gpu": False, "active_gpu": 0})
print("BOOTED", flush=True)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image  # noqa: E402
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


def build_mesh(stage, verts, faces):
    mesh = UsdGeom.Mesh.Define(stage, "/World/Object")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten().astype(np.int32)))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.7, 0.5)])
    return mesh


def intrinsics(stage, w, h):
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
    focal = cam.GetFocalLengthAttr().Get()
    fx = w * focal / cam.GetHorizontalApertureAttr().Get()
    fy = h * focal / cam.GetVerticalApertureAttr().Get()
    return fx, fy


def backproject(depth, fx, fy):
    """z-depth (perpendicular) -> organized camera-frame xyz [H,W,3]; misses (inf) -> 0."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    x = (u - w / 2.0) / fx * z
    y = (v - h / 2.0) / fy * z
    return np.stack([x, y, z], -1).astype(np.float32)


def jitter_pose(mesh, rng, cz):
    """Small pose randomization about the object centre (in-plane spin + tilt + translation)."""
    api = UsdGeom.XformCommonAPI(mesh.GetPrim())
    api.SetRotate(Gf.Vec3f(float(rng.uniform(-8, 8)), float(rng.uniform(-8, 8)),
                           float(rng.uniform(0, 360))))
    api.SetTranslate(Gf.Vec3d(float(rng.uniform(-0.01, 0.01)),
                              float(rng.uniform(-0.01, 0.01)),
                              float(rng.uniform(-0.01, 0.01))))
    _ = cz


def deform(verts, rng, sign):
    """Physical defect: Gaussian z-displacement of a local surface patch (dent sign=+1 pushes the
    surface away from the sensor, bump sign=-1 toward it) — the 3D-mesh analogue of the classical
    normal-displacement defect, but rendered through the path tracer."""
    extent = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    radius = extent * rng.uniform(0.05, 0.12)
    amp = sign * extent * rng.uniform(0.03, 0.06)
    c = verts[rng.randint(len(verts))]
    w = np.exp(-((verts - c) ** 2).sum(1) / (2 * radius ** 2))
    out = verts.copy()
    out[:, 2] += amp * w
    return out, abs(amp)


def render_depth_rgb(rgb_a, dep_a):
    for _ in range(3):
        rep.orchestrator.step(delta_time=0.0, rt_subframes=SUBFRAMES)
    return np.asarray(rgb_a.get_data())[..., :3], np.asarray(dep_a.get_data())


def categories():
    if ARGS.cats:
        return ARGS.cats.split(",")
    return sorted(f[:-4] for f in os.listdir(TWIN_DIR) if f.endswith(".obj"))


s = carb.settings.get_settings()
s.set_bool("/exts/isaacsim.core.throttling/enable_async", False)
s.set_bool("/app/asyncRendering", False)
s.set_int("/omni/replicator/RTSubframes", SUBFRAMES)

rng = np.random.RandomState(0)
timeline = omni.timeline.get_timeline_interface()

for cat in categories():
    verts, faces = parse_obj(f"{TWIN_DIR}/{cat}.obj")
    centroid = verts.mean(0)
    cx, cy, cz = (float(v) for v in centroid)

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    rep.orchestrator.set_capture_on_play(False)

    vc = (verts - centroid).astype(np.float32)             # centred so pose jitter + deform are local
    mesh = build_mesh(stage, vc, faces)
    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Light"))
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Dome"))

    camera = rep.functional.create.camera(position=(0.02, 0.0, -cz), look_at=(0, 0, 0), parent="/World")
    UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera")).CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    rp = rep.create.render_product(camera, (ARGS.res, ARGS.res))
    rgb_a = rep.annotators.get("rgb"); rgb_a.attach(rp)
    dep_a = rep.annotators.get("distance_to_image_plane"); dep_a.attach(rp)
    fx, fy = intrinsics(stage, ARGS.res, ARGS.res)

    def set_points(v, mesh=mesh):
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))

    def randomize_scene(mesh=mesh, cz=cz, light=light, dome=dome):
        jitter_pose(mesh, rng, cz)
        light.CreateIntensityAttr(float(rng.uniform(2000, 4000)))
        dome.CreateIntensityAttr(float(rng.uniform(600, 1400)))

    timeline.play()

    gdir = f"{OUT_ROOT}/{cat}/train/good"
    os.makedirs(f"{gdir}/rgb", exist_ok=True); os.makedirs(f"{gdir}/xyz", exist_ok=True)
    npx = 0
    for i in range(ARGS.views):
        set_points(vc); randomize_scene()
        rgb, depth = render_depth_rgb(rgb_a, dep_a)
        xyz = backproject(depth, fx, fy)
        Image.fromarray(rgb.astype(np.uint8)).save(f"{gdir}/rgb/{i:03d}.png")
        tifffile.imwrite(f"{gdir}/xyz/{i:03d}.tiff", xyz)
        npx += int((np.abs(xyz).sum(-1) > 0).sum())

    n_def = max(1, ARGS.views // 3)
    for dtype, sign in [("dent", 1.0), ("bump", -1.0)]:
        ddir = f"{OUT_ROOT}/{cat}/test/{dtype}"
        for sub in ("rgb", "xyz", "gt"):
            os.makedirs(f"{ddir}/{sub}", exist_ok=True)
        for i in range(n_def):
            set_points(vc); randomize_scene()
            _, good_depth = render_depth_rgb(rgb_a, dep_a)          # baseline at this exact pose
            dverts, amp = deform(vc, rng, sign)
            set_points(dverts)
            rgb, depth = render_depth_rgb(rgb_a, dep_a)             # defect at the same pose
            xyz = backproject(depth, fx, fy)
            obj = np.isfinite(depth) & np.isfinite(good_depth)
            gt = (obj & (np.abs(depth - good_depth) > amp * 0.3)).astype(np.uint8) * 255
            Image.fromarray(rgb.astype(np.uint8)).save(f"{ddir}/rgb/{i:03d}.png")
            tifffile.imwrite(f"{ddir}/xyz/{i:03d}.tiff", xyz)
            Image.fromarray(gt).save(f"{ddir}/gt/{i:03d}.png")

    timeline.stop()
    print(f"CAT {cat:12s} good={ARGS.views} defect={2 * n_def} mean_obj_px={npx // ARGS.views}",
          flush=True)

print("DONE", flush=True)
app.close()
