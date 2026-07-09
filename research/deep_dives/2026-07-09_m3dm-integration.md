# M3DM Integration Plan — learned geometry features to close 0.90→~0.96

**Date**: 2026-07-09
**Status**: partial (integration plan; not yet attempted)
**Supersedes**: none (complements `2026-07-06_mvtec3d-sota.md`, which covered numbers not integration)

## TL;DR

Repo is clean MIT with a modular `Features(nn.Module)` seam that maps onto our `AnomalyMethod` contract — but the **#1 risk is two CUDA-compiled deps (`pointnet2_ops` + `KNN_CUDA`) pinned to torch 1.9/CUDA 11.3 that will not build for Blackwell sm_120** without source patching. Full M3DM is a multi-day fragile-CUDA integration; the **lighter win is extracting Point-MAE per-point features into OUR existing PatchCore bank** (skip UFF fusion + 3-bank machinery), which needs only the geometry backbone and captures most of the hand-FPFH→learned-geometry gain.

## Question

Bead synthscape-iwq: concrete plan to integrate M3DM (learned PointMAE geometry + rgb, multi-memory-bank hybrid fusion) to move PatchCore-rgb (AU-PRO ~0.902) toward the ~0.96 multimodal cluster. Repo/commit/license, weights, sm_120 dep risk, interface seam, compute, and honest effort verdict incl. a lighter alternative.

## Findings

### 1. Repo
- Official: **`github.com/nomewang/M3DM`**, **MIT license** [S1]. (Tencent Youtu mirror `TencentYoutuResearch/AnomalyDetection-M3DM` exists but nomewang is the canonical author repo [S2].)
- **Pin commit `62b7f468aa006f6b348f09e02f85655f416c9d76`** (2023-09-08, latest; repo is dormant since) for `external/` checkout [S3].
- Targets **MVTec 3D-AD specifically** — `utils/preprocessing.py datasets/mvtec3d/`, MVTec-3D is the sole benchmark; it IS the CVPR'23 MVTec-3D SOTA-of-its-era method [S1].

### 2. Pretrained weights (the "learned" features vs our hand-FPFH)
Placed in `checkpoints/` [S1]:
- **Point-MAE** (point-transformer geometry backbone, ShapeNet-pretrained) — Google Drive.
- **DINO ViT-B/8** (rgb backbone) — Google Drive; also supervised IN-1K/IN-21K variants (Google Storage) and Point-BERT (Tsinghua Cloud) as alternates.
- **UFF module** (the unsupervised fusion head) — Google Drive; only needed for `--use_uff` full fusion path.
- Sizes not published in README; ViT-B/8 DINO ≈ 340 MB, Point-MAE ShapeNet ≈ 20–30 MB range (typical). Obtained by manual download (no CLI fetch script) [S1].

### 3. Version-fragile deps — THE integration risk for sm_120 (RTX 5090 / Blackwell)
M3DM's own env is **Ubuntu 18.04 / Python 3.8 / torch 1.9.0 / CUDA 11.3** [S1]. Two deps compile CUDA kernels:
- `pip install .../KNN_CUDA-0.2-py3-none-any.whl` — **KNN_CUDA** [S1]
- `pip install "git+...erikwijmans/Pointnet2_PyTorch...#egg=pointnet2_ops..."` — **pointnet2_ops** (FPS / ball-query / feature interpolation) [S1]

Neither ships sm_120 binaries. On our stack (torch≥2.7 cu130, sm_120) these raise `CUDA error: no kernel image is available for execution on the device` — stable PyTorch wheels were compiled up to sm_90 (Hopper), and any custom CUDA op inherits the same gap [S4]. Known escape hatches, all requiring **build-from-source, not pip**: set `TORCH_CUDA_ARCH_LIST="12.0"` (a.k.a. sm_120) and compile against a CUDA 12.8+/13 toolkit; there is a documented **nvcc `-O3` miscodegen bug for sm_120 → compile the ops at `-O2`** [S4]. No published "known-good M3DM-on-Blackwell" recipe exists; you'd be first, patching `setup.py` of both ops. `mmcv`/`spconv` are NOT required (M3DM avoids them) — the risk is isolated to those two ops. [S1][S4]

### 4. Interface — the seam to our `AnomalyMethod`
- Core is `feature_extractors/features.py::Features(torch.nn.Module)` with `__call__(rgb, xyz)` returning rgb/xyz feature maps + center points + optionally point-interpolated features, and **abstract** `add_sample_to_mem_bank(sample)` / `add_sample_to_late_fusion_mem_bank(sample)` / `predict(sample, mask, label)` [S5]. Concrete subclasses implement the bank + scoring.
- Input: MVTec-3D-style **organized rgb (224×224) + xyz point cloud**; `interpolating_points()` (pointnet2) maps downsampled backbone features back to full point resolution. Output of the pipeline: **per-pixel anomaly maps + image-level score** — the same shape our `ScoreArrays(amaps, valids, masks, scores, labels, defects)` wants [S5].
- Seam verdict: it is **library-shaped, not a monolith** — modular extractors/models/dataset [S1], and the abstract-method pattern is a natural place to wrap. But `main.py` orchestrates fit→save-features→bank→predict as a script; we'd call the `Features` subclass methods directly and adapt its per-category loop into our `fit(cat)->state` / `score(state,cat)` rather than shelling out to `main.py`.

### 5. Compute / data reality
- **Inference memory ~6.52 GB, ~0.51 FPS** with the 3-bank hybrid fusion [S6] — comfortably within the 5090's 32 GB; slow but fine for offline eval. Fit (coreset build over a category's train set) is the heavier RAM/VRAM phase (three banks + coreset subsampling) but MVTec-3D categories are small (~200–300 train samples) — minutes-scale per category on a 5090, not hours.
- Data: M3DM consumes the **raw MVTec-3D organized point clouds** (tiff xyz + rgb), via its own `preprocessing.py`. We have raw at `D:/data/3d`; our processed `s256/s512` stores are our resize — M3DM will want its own preprocessing off the raw, so point at raw, not s256. [S1]

### 6. Honest effort verdict + lighter alternative
- **Full M3DM (DINO+Point_MAE+UFF fusion, 3 banks): multi-day**, dominated by getting `pointnet2_ops`+`KNN_CUDA` to compile for sm_120 (source patch, arch-list, `-O2` nvcc workaround, iterate). Single biggest risk = those two ops; everything else (weights, wrapping the seam) is routine.
- **Lighter alternative (recommended first step): Point-MAE features → OUR existing bank/harness.** Point-MAE backbone itself is plain transformer weights; the pointnet2 dependency is in the *patch grouping / FPS + interpolation* front-end. If we feed Point-MAE our already-sampled patch centers (we control sampling in our preprocessing) we can **sidestep or minimally-port** the pointnet2 front-end and skip `KNN_CUDA` (that's M3DM's Gaussian-blur/bank concern, not the backbone). This drops learned geometry features into the PatchCore bank we already trust, giving most of the hand-FPFH→learned gain for a fraction of the CUDA pain — a plausible **weekend** vs the multi-day full port. Confirm feasibility by reading `models/pointnet2_utils.py` grouping once checked out.

## Open questions

- Exact checkpoint file sizes + whether Google Drive links still resolve (repo dormant since 2023 — verify at checkout).
- Our s256/s512 resize degrades geometry (noted in `2026-07-06_mvtec3d-sota.md`); M3DM off raw may itself recover AU-PRO independent of the learned backbone — worth an ablation to attribute the gain.

## RESOLVED at checkout (2026-07-09, code read @62b7f46)

- **Can Point-MAE run with OUR FPS/grouping, avoiding pointnet2_ops? YES.** `models.Group.forward` does exactly two CUDA-op calls: `fps()` (`pointnet2_ops.furthest_point_sample` + `gather_operation`) and `self.knn` (`knn_cuda.KNN`). Both have **pure-torch references already in the repo** (`models/pointnet2_utils.py`: `farthest_point_sample`, `square_distance`, `index_points`). Reimplemented as `surfscan/models/pointmae_group.py` (100% tested vs brute force) — Point-MAE grouping now runs with **zero custom CUDA ops** → the #1 sm_120 risk is avoided for the lite path.
- **Does KNN_CUDA sit on the feature path or only the blur? Both — but the feature-path use is replaced.** `knn_cuda.KNN` is used in `models.Group` (grouping, now replaced above) AND in `features.py` only via `KNNGaussianBlur` (anomaly-map post-smoothing, droppable / pure-torch-able). `interpolating_points` (features.py) is already pure-torch. So after the grouping swap, **no CUDA op remains on the feature path.**
- Remaining lite-path work (GPU-gated): load Point-MAE ShapeNet weights, wire `Group`→our `pointmae_group`, extract per-point features → our PatchCore bank (`coreset.py`/`bank_nn_dist`) as a `PointMAEMethod(AnomalyMethod)`, fit/score one category.

## Sources

- [S1] nomewang/M3DM README — github.com/nomewang/M3DM — accessed 2026-07-09 (license, deps, install cmds, weights, run cmds, dataset)
- [S2] WebSearch: M3DM repo / Tencent Youtu mirror — accessed 2026-07-09
- [S3] GitHub API commits, nomewang/M3DM — commit 62b7f468 dated 2023-09-08 — accessed 2026-07-09
- [S4] sm_120/Blackwell PyTorch + custom-CUDA-op build issues — NVIDIA Dev Forums #338015, pytorch/pytorch#159207, sm_120 nvcc -O3 miscodegen note — accessed 2026-07-09
- [S5] feature_extractors/features.py @62b7f468 — `Features(nn.Module)`, `__call__(rgb,xyz)`, abstract add_sample_to_mem_bank/predict, interpolating_points — accessed 2026-07-09
- [S6] M3DM inference cost 6.52 GB / 0.51 FPS, 3 memory banks — Memoryless-Multimodal (arXiv 2409.05378) comparison — accessed 2026-07-09
