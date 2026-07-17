"""Look at ONE MVTec 3D-AD sample — the "see the data" step.

The whole point of running this: feel in your hands that a "sample" is a
*synced grid* of two images on the same pixel coordinates —
    rgb/NNN.png   -> (r,g,b) per pixel   (the photo)
    xyz/NNN.tiff  -> (x,y,z) per pixel   (the surface geometry, already un-projected)
    gt/NNN.png    -> defect mask         (0 = normal, >0 = defect; all-black for 'good')
Lift the grid to points, paint with the photo, spin it. That's 2.5D.

Run (after `uv sync`):
    uv run python -m surfscan.viz show --cat bagel --split test --defect hole --idx 0 --mask
    uv run python -m surfscan.viz show --cat bagel --split train --defect good --idx 0   # a normal one

Opens the Open3D 3D window straight away (drag = rotate, scroll = zoom,
right-drag = pan, q = quit); the defect is painted red with --mask. Add
--rgb to also pop the flat photo + mask first (matplotlib, blocks until
closed). --no-3d with --rgb = photo only.
"""
import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tifffile
import torch
from PIL import Image

from core import config
from core.compute import Compute
from core.data.static import preprocess as pp
from core.data.static import store
from core.data.static.dataset import GpuSplit
from core.data.static.mvtec import Split
from core.obs import Obs
from surfscan.dispatch import Spec
from surfscan.models.patchcore import FitCfg, PatchCore
from surfscan.models.vae import ConvVAE
from surfscan.training.hparams import HParams

log = Obs.get()

_RAW_RGB_CUTOFF = 1.5   # cols.max() above this => raw uint8 (not float [0,1])


class Show:
    """Look at ONE MVTec 3D-AD sample — the 'see the data' step (load, lift to points, recolor, spin).

    The loaded sample (rgb + xyz + valid + gt) is the object's state: every render/colorize method
    operates on that same synced grid. Loaders PRODUCE a sample and stay static; the CLI entry builds
    one `Show(...)` and renders it.
    """

    def __init__(self, rgb, xyz, valid, gt=None):
        self.rgb = rgb
        self.xyz = xyz
        self.valid = valid
        self.gt = gt

    @staticmethod
    def load_sample(root, cat, split, defect, idx):
        base = Path(root) / cat / split / defect
        stem = f"{idx:03d}"
        rgb = np.asarray(Image.open(base / "rgb" / f"{stem}.png"))      # H x W x 3, uint8
        xyz = tifffile.imread(base / "xyz" / f"{stem}.tiff")            # H x W x 3, float32
        gt_path = base / "gt" / f"{stem}.png"
        gt = np.asarray(Image.open(gt_path)) if gt_path.exists() else None
        return rgb, xyz, gt

    @staticmethod
    def load_processed(cat, split, defect, idx, size=None):
        """Load one CLEANED sample from the processed store (rgb [0,1], xyz normalized,
        valid + gt)."""
        sid = f"{cat}_{split}_{defect}_{idx:03d}"
        path = store.Store.dataset_dir(size=size or pp.SIZE) / "data" / f"{sid}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not built — run: python -m core.data.static.store --cats {cat}")
        a = store.Store.load_arrays(path)
        return a["rgb"], a["xyz"], a["gt"], a["valid"]

    def _run_model(self, run, device=None):
        """Load a trained model + run it on this sample -> (input x [1,C,H,W], recon [1,C,H,W]), numpy."""
        run = Path(run)
        hp = HParams(**json.loads((run / "config.json").read_text()))
        chans = [(self.xyz if c == "xyz" else self.rgb).transpose(2, 0, 1) for c in hp.channels]
        x = np.concatenate(chans, 0)[None].astype("float32")          # 1,C,H,W
        dev = device or Compute.pick_device()
        model = ConvVAE(hp.model_cfg(x.shape[1])).to(dev)
        model.load_state_dict(torch.load(run / "model.pt", map_location=dev))
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(torch.from_numpy(x).to(dev))
        return x, recon.cpu().numpy()

    def anomaly_map(self, run):
        """Per-pixel reconstruction error from a trained model (run dir). Requires processed inputs."""
        x, recon = self._run_model(run)
        err = ((recon - x) ** 2).sum(1).squeeze(0)                    # H,W
        return err * self.valid

    def recon_xyz(self, run):
        """The model's reconstructed xyz channels (H,W,3) — what it rebuilt the surface as."""
        _, recon = self._run_model(run)
        return recon[0, :3].transpose(1, 2, 0)                        # C,H,W -> H,W,3 (xyz channels)

    def patchcore_map(self, cat, coreset=0.1, device=None):
        """The WORKING detector's anomaly map: fit a PatchCore bank on this category's train-good (rgb),
        score this sample. The defect should glow (unlike the VAE's anti-localized residual)."""
        dev = device or Compute.pick_device()
        train = GpuSplit.load_split(split=Split.TRAIN, label=0, cats=[cat], channels=["rgb"], device=dev)
        pc = PatchCore(device=dev).fit(train.x, FitCfg(coreset=coreset))
        x = torch.from_numpy(self.rgb.transpose(2, 0, 1)[None]).to(dev)
        return pc.score_maps(x)[0] * self.valid                       # H,W

    def to_point_cloud(self, *, mask=False):
        """Drop background, return (points Nx3, colors Nx3 in [0,1]).

        rgb may be uint8 (raw) or float [0,1] (processed) — normalized either way.
        With mask, the labelled gt defect region is painted red.
        """
        gt = self.gt if mask else None
        pts = self.xyz[self.valid]
        cols = self.rgb[self.valid].astype(np.float64)
        if cols.size and cols.max() > _RAW_RGB_CUTOFF:          # raw uint8 -> [0,1]
            cols /= 255.0
        if gt is not None:                          # paint the labelled defect region red
            cols = cols.copy()
            cols[gt[self.valid] > 0] = (1.0, 0.0, 0.0)
        return pts, cols

    def _show_photo(self, args):
        cols_n = 1 if self.gt is None else 2
        fig, axes = plt.subplots(1, cols_n, figsize=(5 * cols_n, 5))
        axes = np.atleast_1d(axes)
        axes[0].imshow(self.rgb)
        axes[0].set_title(f"{args.cat}/{args.split}/{args.defect}/{args.idx:03d} — RGB")
        axes[0].axis("off")
        if self.gt is not None:
            axes[1].imshow(self.gt, cmap="gray")
            axes[1].set_title("GT defect mask")
            axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    def _colorize(self, pc, args):
        """Recolor the cloud by the chosen mode (anomaly / curvature / normals); returns show_normals."""
        if args.anomaly is not None or args.patchcore:
            if not args.processed:
                raise SystemExit("--anomaly/--patchcore need --processed (the model eats the processed input)")
            err = (self.patchcore_map(args.cat) if args.patchcore
                   else self.anomaly_map(args.anomaly))
            e = err[self.valid.astype(bool)]
            e = (e / (np.percentile(e, 99) + 1e-9)).clip(0, 1)
            pc.colors = o3d.utility.Vector3dVector(cm.inferno(e)[:, :3])
            tag = "patchcore" if args.patchcore else "recon residual"
            log.info(f"{tag}: median {np.median(err[self.valid.astype(bool)]):.4f}  "
                     f"p99 {np.percentile(err[self.valid.astype(bool)], 99):.4f}")
        elif args.curvature:
            pc.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
            ev = np.linalg.eigvalsh(np.asarray(pc.covariances))
            sv = ev[:, 0] / (ev.sum(1) + 1e-12)
            sv_n = (sv / (np.percentile(sv, 99) + 1e-12)).clip(0, 1)
            pc.colors = o3d.utility.Vector3dVector(cm.inferno(sv_n)[:, :3])
            log.info(f"surface variation: median {np.median(sv):.4f}  p99 {np.percentile(sv, 99):.4f}")
        elif args.normals:
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
            pc.orient_normals_towards_camera_location((0.0, 0.0, 0.0))
            if not args.mask:
                n = np.asarray(pc.normals)
                pc.colors = o3d.utility.Vector3dVector((n * 0.5 + 0.5).clip(0, 1))
            return args.arrows
        return False

    @staticmethod
    def args(ap):
        ap.add_argument("--data-root", default=None,
                        help="MVTec 3D-AD root (default: <paths.yaml data>/raw/mvtec_3d_anomaly_detection)")
        ap.add_argument("--cat", default="bagel")
        ap.add_argument("--split", default=Split.TEST, help="train | validation | test")
        ap.add_argument("--defect", default="hole", help="good | crack | hole | contamination | combined | ...")
        ap.add_argument("--idx", type=int, default=0)
        ap.add_argument("--mask", action="store_true", help="paint the GT defect region red in 3D")
        ap.add_argument("--rgb", action="store_true", help="also show the flat RGB photo + GT mask (matplotlib)")
        ap.add_argument("--no-3d", action="store_true", help="skip the 3D window (use with --rgb for photo only)")
        ap.add_argument("--processed", action="store_true",
                        help="render the CLEANED processed sample (post background/outlier/normalize) instead of raw")
        ap.add_argument("--size", type=int, default=None, help="processed param_key size (default: preprocess.SIZE)")
        ap.add_argument("--normals", action="store_true",
                        help="color points by surface-normal direction (the surface bend, made visible) — "
                             "see L2 of learning/2026-06-25_fpfh-and-neighborhoods.md")
        ap.add_argument("--arrows", action="store_true",
                        help="with --normals: also draw the normal vectors as arrows (50k of them — noisy; opt-in)")
        ap.add_argument("--curvature", action="store_true",
                        help="color points by surface variation lambda2/(sum) — flat=dark, edges/noise/defects "
                             "glow. The geometric-weirdness score, a baby anomaly heatmap.")
        ap.add_argument("--anomaly", type=Path, default=None,
                        help="run dir (e.g. runs/vae_all/bagel): color points by the model's reconstruction "
                             "anomaly score (inferno). Requires --processed.")
        ap.add_argument("--patchcore", action="store_true",
                        help="color points by the WORKING detector (PatchCore, fit on this category's "
                             "train-good) — the defect glows. Requires --processed.")
        ap.add_argument("--recon", type=Path, default=None,
                        help="run dir: show the model's RECONSTRUCTED surface (its rebuilt xyz), colored by the "
                             "original rgb. See what it actually rebuilds. Requires --processed.")

    @staticmethod
    def run(args):
        if args.processed:
            rgb, xyz, gt, valid = Show.load_processed(args.cat, args.split, args.defect, args.idx, args.size)
        else:
            root = args.data_root or config.Config.mvtec_root()
            rgb, xyz, gt = Show.load_sample(root, args.cat, args.split, args.defect, args.idx)
            valid = np.any(xyz != 0, axis=-1)
        show = Show(rgb, xyz, valid, gt)
        log.info(f"rgb {rgb.shape} {rgb.dtype} | xyz {xyz.shape} {xyz.dtype} | "
              f"gt {None if gt is None else gt.shape}")
        log.info(f"object pixels: {valid.sum():,} / {valid.size:,}  ({100 * valid.mean():.1f}%)")
        if gt is not None:
            log.info(f"defect pixels: {(gt > 0).sum():,}  ({100 * (gt > 0).mean():.2f}% of frame)")

        if args.rgb:                                     # opt-in flat photo + mask (blocks until closed)
            show._show_photo(args)

        if args.no_3d:
            return

        # 2) the colored 3D point cloud — same grid, now as geometry
        if args.recon is not None:
            if not args.processed:
                raise SystemExit("--recon needs --processed (the model eats the normalized processed input)")
            show.xyz = show.recon_xyz(args.recon)        # show the rebuilt surface instead of the input
            log.info("showing the model's RECONSTRUCTED surface (colored by original rgb)")
        pts, cols = show.to_point_cloud(mask=args.mask)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(cols)

        show_normals = show._colorize(pc, args)

        log.info("3D: drag = rotate, scroll = zoom, q = quit")
        o3d.visualization.draw_geometries([pc], window_name="MVTec 3D-AD sample",
                                          point_show_normal=show_normals)


SPEC = Spec("show", Show.args, Show.run)
