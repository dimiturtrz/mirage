# MVTec 3D-AD SOTA — current landscape (verified 2026-07-06)

Owed ⚠-verify pass: our docs cited M3DM (~0.96, CVPR 2023) as "SOTA," which went stale. This settles
the current top from a primary comparison table.

## Verified numbers (from DCRDF-Net's comparison table, PMC12846035, 2026)
Mean over the 10 categories, RGB+3D:

| method | year | image I-AUROC | pixel AU-PRO |
|---|---|---|---|
| **DCRDF-Net** (current top) | 2026 | **0.971** | **0.988** |
| MMRD | 2025 | 0.950 | 0.976 |
| Shape-Guided | 2023 | 0.947 | 0.976 |
| M3DM (the classic ref) | 2023 | 0.945 | 0.964 |
| AST | 2023 | 0.937 | 0.944 |
| BTF | 2023 | 0.865 | 0.959 |
| EasyNet | 2023 | 0.926 | 0.919 |

So the current top **cluster** is ~0.95–0.97 I-AUROC / ~0.96–0.99 AU-PRO; DCRDF-Net reports the highest
on both. M3DM (our old "SOTA") is now mid-pack.

## Where mirage sits
Our best (our harness, rgb-only PatchCore / feature-recon): **0.819 I-AUROC · 0.908 AU-PRO.**
- Gap to current top (DCRDF-Net): **~15 pp** image-AUROC, **~8 pp** AU-PRO.
- The **detection (image-AUROC) gap is the bigger one** — our image scoring is weak on hard categories.
- Caveat: our numbers are *our* eval protocol (self-verified perfect=1.0/random=0.17), not the official
  released-code setting — cross-comparison is approximate, not a leaderboard entry.

## The through-line
Every top method is **multimodal RGB+3D fusion** (M3DM hybrid fusion, MMRD/Shape-Guided/DCRDF geometry+
appearance). Our gap **is** that fusion — we're rgb-only + random coreset + geometry we degrade in
preprocessing (grazing-noise / 256-resize). Named + measured, consistent with RESULTS.md's caveats.

## Honesty notes
- "SOTA" is contested/shifting; single-paper "surpasses SOTA" claims are self-serving. Framed as
  "current top cluster," DCRDF-Net cited as the highest reported, M3DM as the well-established reference.
- Not conflated: **TransFusion's 0.992 is on MVTec AD (2D)**, not MVTec 3D-AD — a different benchmark.
- Per PLAN rule #1 we don't chase this leaderboard; the contribution is eval rigor + the sim-to-real
  engine. Being ~8 pp back on a benchmark we don't optimize is expected and stated.

Sources: [DCRDF-Net (PMC12846035)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12846035/) ·
[M3DM CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.pdf) ·
[G2SF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Tao_G2SF_Geometry-Guided_Score_Fusion_for_Multimodal_Industrial_Anomaly_Detection_ICCV_2025_paper.pdf)
