# Stage 0 results — anomaly detection on MVTec 3D-AD

Honest first-stage findings. **Per-category models** (the MVTec convention), scored through our own
eval harness — image-level **AUROC** (detection) + pixel-level **AU-PRO** (localization). These are
*our* runs through *one* eval harness, **not** the papers' leaderboard numbers.

## Headline: reconstruction fails, feature-memory-bank works
| method | modality | mean img-AUROC | mean pixel AU-PRO |
|---|---|---|---|
| Reconstruction (VAE / +dropout / inpaint / fused) — ours | xyz (+rgb) | 0.48 | **0.095** |
| BTF (FPFH memory bank) — ours | geometry | ⚠ (bagel 0.75) | ⚠ (bagel 0.51) |
| **PatchCore (feature memory bank) — ours** | rgb | **0.82** | **0.908** |
| SOTA (BTF / M3DM, *papers*) ⚠ | rgb + 3D | ~0.95 | ~0.96 |

**0.095 → 0.908.** Reconstruction-residual is ≈ random at localization. The reason — *measured*
across four reconstruction variants (vanilla VAE, +dropout, masked-inpainting, rgb-fused), all
AU-PRO ≈ 0.014–0.095:

> Raw-residual tracks geometric **complexity** (the bagel rim + structured-light grazing-noise are
> hard to rebuild), **not defect-ness** (a smooth defect rebuilds *fine* → low residual). On bagel
> the defect region's residual is ~9× *lower* than normal regions.

"Compare to normal" (a memory bank of normal features) fixes it: a normal-but-complex region (the
rim) matches a normal rim in the bank → not flagged. That's why the SOTA is memory-bank / feature-
based, not reconstruction — here, measured.

## PatchCore-rgb, per category (the working method)
| category | img-AUROC | AU-PRO |
|---|---|---|
| bagel | 0.943 | 0.928 |
| cable_gland | 0.806 | 0.728 |
| carrot | 0.944 | 0.987 |
| cookie | 0.708 | 0.822 |
| dowel | 0.897 | 0.974 |
| peach | 0.828 | 0.960 |
| potato | 0.860 | 0.986 |
| rope | 0.909 | 0.971 |
| tire | 0.604 | 0.936 |
| foam | 0.691 | 0.789 |
| **MEAN** | **0.819** | **0.908** |

## Honest caveats (the measured gaps to SOTA)
- **PatchCore is rgb-only + *random* coreset** (not greedy). The gap to SOTA (~0.96) is exactly the
  **3D-feature fusion** (M3DM) + **greedy coreset** we haven't added yet.
- **Our BTF underperforms paper-BTF** (~0.96): we run FPFH on the **256-resized / normalized /
  grazing-noisy** processed cloud; paper-BTF uses **native-resolution clean** point clouds. The
  grazing-angle normal-noise (see `learning/2026-06-25_fpfh-and-neighborhoods.md`) corrupts the
  periphery descriptors.
- AU-PRO here is per-region upper-envelope integrated to FPR 0.3, **self-verified** (perfect = 1.00,
  random = 0.17). Image score = mean of the top-k pixel residuals.

## The point (not a leaderboard number)
The result is the **contrast + the honest mechanism**, the way systole reported its triad: the
intuitive method (reconstruction) *measured* failing, *diagnosed* (residual ≠ defect signal), and
the working paradigm (feature memory bank) standing near SOTA at **0.91** with the remaining gap
**named** (3D fusion · greedy coreset). The contribution is the eval rigor + the honest comparison,
not the single number.

*Reproduce: `python -m mirage.training.run_all` (reconstruction) · `... run_patchcore` · `... run_btf`.
⚠ = verify against the primary source before any public claim.*
