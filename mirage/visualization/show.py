"""Look at ONE MVTec 3D-AD sample — the "see the data" step.

The whole point of running this: feel in your hands that a "sample" is a
*synced grid* of two images on the same pixel coordinates —
    rgb/NNN.png   -> (r,g,b) per pixel   (the photo)
    xyz/NNN.tiff  -> (x,y,z) per pixel   (the surface geometry, already un-projected)
    gt/NNN.png    -> defect mask         (0 = normal, >0 = defect; all-black for 'good')
Lift the grid to points, paint with the photo, spin it. That's 2.5D.

Run (in the `mirage` conda env, after `pip install -e .`):
    python -m mirage.visualization.show --cat bagel --split test --defect hole --idx 0 --mask
    python -m mirage.visualization.show --cat bagel --split train --defect good --idx 0   # a normal one

Opens the Open3D 3D window straight away (drag = rotate, scroll = zoom,
right-drag = pan, q = quit); the defect is painted red with --mask. Add
--rgb to also pop the flat photo + mask first (matplotlib, blocks until
closed). --no-3d with --rgb = photo only.
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from mirage import config


def load_sample(root, cat, split, defect, idx):
    base = Path(root) / cat / split / defect
    stem = f"{idx:03d}"
    rgb = np.asarray(Image.open(base / "rgb" / f"{stem}.png"))      # H x W x 3, uint8
    xyz = tifffile.imread(base / "xyz" / f"{stem}.tiff")            # H x W x 3, float32
    gt_path = base / "gt" / f"{stem}.png"
    gt = np.asarray(Image.open(gt_path)) if gt_path.exists() else None
    return rgb, xyz, gt


def to_point_cloud(rgb, xyz, gt=None):
    """Drop background, return (points Nx3, colors Nx3 in [0,1])."""
    valid = np.any(xyz != 0, axis=-1)          # background pixels are exactly (0,0,0)
    pts = xyz[valid]
    cols = rgb[valid].astype(np.float64) / 255.0
    if gt is not None:                          # paint the labelled defect region red
        cols[gt[valid] > 0] = (1.0, 0.0, 0.0)
    return pts, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="MVTec 3D-AD root (default: <paths.yaml data>/raw/mvtec_3d_anomaly_detection)")
    ap.add_argument("--cat", default="bagel")
    ap.add_argument("--split", default="test", help="train | validation | test")
    ap.add_argument("--defect", default="hole", help="good | crack | hole | contamination | combined | ...")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--mask", action="store_true", help="paint the GT defect region red in 3D")
    ap.add_argument("--rgb", action="store_true", help="also show the flat RGB photo + GT mask (matplotlib)")
    ap.add_argument("--no-3d", action="store_true", help="skip the 3D window (use with --rgb for photo only)")
    args = ap.parse_args()

    root = args.data_root or config.mvtec_root()
    rgb, xyz, gt = load_sample(root, args.cat, args.split, args.defect, args.idx)
    valid = np.any(xyz != 0, axis=-1)
    print(f"rgb {rgb.shape} {rgb.dtype} | xyz {xyz.shape} {xyz.dtype} | "
          f"gt {None if gt is None else gt.shape}")
    print(f"object pixels: {valid.sum():,} / {valid.size:,}  ({100 * valid.mean():.1f}%)")
    if gt is not None:
        print(f"defect pixels: {(gt > 0).sum():,}  ({100 * (gt > 0).mean():.2f}% of frame)")

    # 1) (optional) the flat photo + mask — image-land. Opt-in via --rgb; it blocks
    #    until closed, so default is straight to the 3D cloud (which is already
    #    RGB-colored, and --mask shows the defect there too).
    if args.rgb:
        import matplotlib.pyplot as plt
        cols_n = 1 if gt is None else 2
        fig, axes = plt.subplots(1, cols_n, figsize=(5 * cols_n, 5))
        axes = np.atleast_1d(axes)
        axes[0].imshow(rgb)
        axes[0].set_title(f"{args.cat}/{args.split}/{args.defect}/{args.idx:03d} — RGB")
        axes[0].axis("off")
        if gt is not None:
            axes[1].imshow(gt, cmap="gray")
            axes[1].set_title("GT defect mask")
            axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    if args.no_3d:
        return

    # 2) the colored 3D point cloud — same grid, now as geometry
    import open3d as o3d
    pts, cols = to_point_cloud(rgb, xyz, gt if args.mask else None)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(cols)
    print("3D: drag = rotate, scroll = zoom, q = quit")
    o3d.visualization.draw_geometries([pc], window_name="MVTec 3D-AD sample")


if __name__ == "__main__":
    main()
