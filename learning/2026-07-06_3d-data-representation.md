# 3D data representation — position maps, point clouds, 2.5D (the foundation)

The concept everything sits on: *what is the data, actually?* (CS231A week 1 for our case.) Learned
after FPFH by accident — this is the ground under it.

## Three containers for visual data
- **Image** — a grid of pixels, each holding **colour** `(r,g,b)`. Pure appearance, 2D. No geometry.
- **Point cloud** — an **unordered set** of 3D points `{(x,y,z)}`. Pure geometry, no grid, no neighbours-
  by-index. What FPFH/point-transformers consume.
- **Position map (a.k.a. organized point cloud / XYZ image / 2.5D)** — a **grid, image-shaped**, where each
  pixel holds a **3D coordinate** `(x,y,z)` instead of a colour. Geometry *arranged on a grid.* Best of
  both: you can treat it as an image (2D CNN) **or** lift it to a point cloud.

## What MVTec actually gives — a position map
Per sample, an **800×800 grid**, three **synced** channels on the *same pixel coordinates*:
- `rgb/NNN.png` → `(r,g,b)` — the photo (appearance).
- `xyz/NNN.tiff` → `(x,y,z)` — the 3D coordinate, in **metres**, of the surface point that pixel saw
  (from the **Zivid structured-light scanner**). This IS a point cloud, just laid on a grid.
- `gt/NNN.png` → the defect mask (test split only).

Because the channels share pixel coordinates, `rgb[i,j]` and `xyz[i,j]` describe the *same physical point*
— appearance and geometry are pixel-aligned for free.

## Why "2.5D," not "3D"
It's a **single view** — you get the surface *facing the scanner* (a height-field / depth from one
camera), **not the full 360° solid.** Front yes, back no. So: a 2D grid + one depth per pixel = a
*surface*, not a volume. Hence "2.5-dimensional." (This is exactly why the digital-twin reconstruction —
completing the unseen back — is hard: the data only ever saw one side.)

## Already un-projected (the subtle bit)
A plain **depth image** stores only `z` (distance) per pixel; to get real `(x,y,z)` you'd need the
**camera intrinsics** to un-project. MVTec's XYZ tiff **already stores full `(x,y,z)` per pixel** — the
scanner did the un-projection. So we can lift straight to a point cloud with no camera math: every valid
pixel *is* a 3D point.

## The background / valid mask
In the position map, **background pixels are exactly `(0,0,0)`** — the scanner returns 0 where no surface
was hit. That's the **only** background marker (rgb doesn't encode it). So: **object = pixels where
`xyz ≠ 0`.** Preprocessing = that mask + flying-pixel removal → the object surface. (This is why
`preprocess.py` keys off `xyz != 0`.)

## Two ways we use it (the fork the whole repo runs on)
- **Keep the grid** (position map → 2D CNN, channels = xyz): plays the CNN strength, per-pixel structure
  preserved, per-pixel anomaly map for free. What the VAE/PatchCore paths do.
- **Drop the grid** (lift valid pixels → unordered point cloud): feed geometry methods (FPFH normals,
  point transformers). What BTF does. Loses grid-neighbour structure, gains true-3D operations.

---
## Quiz log

### 2026-07-06 — quiz 1 · score ~3.1/6 (52%)
- **Q1 containers** ✅1.0 — rgb 3, depth 1, position map 3=(x,y,z), cloud=set of points, "semantics downstream." Solid.
- **Q2 2.5D** ⚠️0.5 — right conclusion ("no whole surface graph") but WRONG mechanism: said "snapshots from different sides." It's a SINGLE view; 2.5D = 2D grid + 1 depth = a front *surface*, not sides aggregated.
- **Q3 un-projection** ⚠️0.4 — "translation to a coordinate system," vague. Missed: position map skips the CAMERA INTRINSICS + un-projection math a depth image needs.
- **Q4 shape from where** ⚠️0.4 — "sampling" ✅ then drifted to "aggregate many samplings." Missed: shape is INFERRED from DENSE surface samples (one view already looks like a bagel), not stored, not aggregated.
- **Q5 the fork** ❌0 — didn't recall. keep-grid→2D CNN (per-pixel map free) vs drop-grid→point cloud (true-3D, but build kd-tree).
- **Q6 spot the error** ✅0.8 — recalled point cloud ≠ world model; world model is downstream fusion.

**Recurring error (the fix):** keeps importing "aggregate many views → build a model" (that's Stage-2
reconstruction) into the SINGLE-view basics (cost Q2, Q4). Burn in: *our data = ONE view → one position
map → one front-surface cloud. No aggregation, no world model — that's later, and we don't have it.*
Weak spots to re-lock: single-view 2.5D · un-projection (no intrinsics) · the grid/point-cloud fork.
