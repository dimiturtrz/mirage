"""Raw sample -> the unified processed tensor (the preprocess contract, docs/PLAN.md).

Steps: background mask (xyz == 0) -> flying-pixel removal (statistical outlier) ->
crop to object bbox -> resize to a fixed grid -> normalize geometry to a known
physical scale (center on the valid centroid, divide by a fixed constant, NOT
dataset statistics) + rgb/255. Invalid pixels are zeroed. Returns the arrays
cached as one npz per sample.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion
from skimage.transform import resize as sk_resize

# default preprocess params -> param_key, so recipes never collide in processed/
SIZE = 256          # output grid (H = W = SIZE)
_MASK_BIN = 0.5     # re-binarize a mask after float resize
SO_NB = 20          # statistical-outlier removal: neighbor count
SO_STD = 2.0        # statistical-outlier removal: std-ratio threshold
SCALE = 0.05        # geometry scale in meters (fixed/physical, NOT dataset stats)
CLIP = 4.0          # drop points beyond this radius (scaled units) from the surface centroid
                    # — kills the flying-pixel *cluster* SOR can't (its points are each other's neighbors)


def _remove_flying_pixels(xyz, valid, nb=SO_NB, std=SO_STD):
    """Drop structured-light flying pixels (non-zero outliers at depth edges)."""
    pts = xyz[valid]
    if len(pts) < nb + 1:
        return valid
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    _, keep = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    out = np.zeros_like(valid)
    coords = np.argwhere(valid)
    out[tuple(coords[keep].T)] = True
    return out


def _bbox(valid):
    ys, xs = np.where(valid)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def preprocess(rgb, xyz, gt, size=SIZE, nb=SO_NB, std=SO_STD, scale=SCALE, clip=CLIP):
    valid = np.any(xyz != 0, axis=-1)                 # background = exactly (0,0,0)
    valid = _remove_flying_pixels(xyz, valid, nb, std)
    if gt is None:
        gt = np.zeros(valid.shape, np.uint8)

    y0, y1, x0, x1 = _bbox(valid)
    rgb, xyz, gt, valid = (a[y0:y1, x0:x1] for a in (rgb, xyz, gt, valid))

    # resize to a fixed square grid (rgb/xyz bilinear; mask/gt nearest)
    rgb = sk_resize(rgb, (size, size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32) / 255.0
    xyz = sk_resize(xyz, (size, size), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
    valid = sk_resize(valid.astype(np.float32), (size, size), order=0, preserve_range=True) > _MASK_BIN
    # erode 1px: bilinear xyz blends object-edge pixels with (0,0,0) background, corrupting their
    # geometry toward the origin; drop that boundary ring so only un-blended interior points remain.
    valid = binary_erosion(valid, iterations=1)
    gt = (sk_resize(gt.astype(np.float32), (size, size), order=0, preserve_range=True) > _MASK_BIN).astype(np.uint8)
    gt = (gt & valid).astype(np.uint8)

    # geometry -> known physical scale: center on valid centroid, divide by fixed constant
    centroid = xyz[valid].mean(0) if valid.any() else np.zeros(3, np.float32)
    xyz = ((xyz - centroid) / scale).astype(np.float32)

    # drop the flying-pixel cluster: anything beyond `clip` of the centroid isn't the surface
    valid = valid & (np.linalg.norm(xyz, axis=-1) < clip)

    m = valid[..., None]
    xyz = (xyz * m).astype(np.float32)
    rgb = (rgb * m).astype(np.float32)
    gt = (gt & valid).astype(np.uint8)

    return {"xyz": xyz, "rgb": rgb, "valid": valid, "gt": gt}
