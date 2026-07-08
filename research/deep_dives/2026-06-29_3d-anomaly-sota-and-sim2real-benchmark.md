# 3D-anomaly SOTA number + sim-to-real benchmark — pre-public claim verification

**Date:** 2026-06-29 · **Why:** two README claims needed a primary source before the public flip
(integrity rule: verify every strong claim). Both checked.

## Claim 1 — "SOTA (M3DM) ~0.96 pixel AU-PRO" → ✓ STANDS
M3DM (*Multimodal Industrial Anomaly Detection via Hybrid Fusion*, Wang et al., **CVPR 2023**) on
MVTec 3D-AD (10 categories, RGB + point cloud):
- **Pixel AU-PRO (3D+RGB): 0.964** — this is the number the README table cites as "~0.96". Correct.
- Image-AUROC (3D+RGB): 0.945. Pure-3D (no RGB) image-AUROC: 0.874.

Earlier worry that "~0.96" was really 0.945 was a **conflation of image-AUROC with pixel AU-PRO** —
they are different metrics. The README's AU-PRO column is right. No change needed.

Sources:
- [M3DM, CVPR 2023 (CVF open access)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.pdf)
- [alphaXiv overview](https://www.alphaxiv.org/overview/2303.00601v2)

## Claim 2 — "No standardized sim-to-real 3D-anomaly benchmark exists" → ✗ OUTDATED, FIXED
**SiM3D** (*Single-instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark*,
arXiv **2506.21549**, 2025) is exactly such a benchmark:
- "**First to feature single-instance training and explicitly evaluate generalisation from
  synthetic CAD data to real objects (synth2real setup).**"
- 333 instances, 8 manufactured object types, high-res scans; task = voxel-based Anomaly Volume
  (3D localization, not 2D maps).
- Positioned as building on the MVTec family (MVTec 3D-AD, Eyecandies, PAD, Real-IAD, Real3D-AD),
  adding the synth→real generalization challenge those didn't jointly tackle.

So the flat "no benchmark exists" claim is false as written. **Fixed** in README + PLAN: cite SiM3D,
reframe the opening as *barely-charted* and position mirage's contribution as the **physics-based
defect-generation engine + closed-loop curriculum + measured gap on commodity MVTec-style
scans** — complementary to SiM3D (CAD-based detection benchmark), not pre-empted by it.

Sources:
- [SiM3D, arXiv 2506.21549](https://arxiv.org/abs/2506.21549)
- [SiM3D HTML](https://arxiv.org/html/2506.21549)

## Residual
The other ⚠️ markers in `docs/PLAN.md` "SOTA / OSS landscape" (BTF/PatchCore-3D exact numbers, AV
sim-to-real DA %s) are not in the README yet — owed only if/when they migrate there.
