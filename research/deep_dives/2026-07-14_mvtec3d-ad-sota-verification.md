# MVTec 3D-AD Anomaly Detection SOTA Verification

**Date**: 2026-07-14
**Status**: settled
**Supersedes**: none

## TL;DR

DCRDF-Net (published Jan 2026 in Sensors) achieves the highest reported metrics on MVTec 3D-AD: 97.1% image-level I-AUROC and 98.8% pixel-level PRO, surpassing recent SOTA methods. DCRDF-Net is a published, peer-reviewed paper with verifiable metrics. Top 5 methods span 2023–2026 and cluster in 94–97% I-AUROC range; Shape-Guided and M3DM are strong 2023 baselines; CPMF ranks second by image-AUROC (95.15%).

## Question

1. Does DCRDF-Net exist as a published paper with reported MVTec 3D-AD metrics?
2. What are the top ~5 SOTA methods on MVTec 3D-AD by mean AU-PRO and mean image-AUROC, with verified metrics?

## Findings

### DCRDF-Net Verification

**YES — DCRDF-Net is a published, peer-reviewed paper.**

- **Full Title**: "DCRDF-Net: A Dual-Channel Reverse-Distillation Fusion Network for 3D Industrial Anomaly Detection" [S1]
- **Publication**: Sensors, Volume 26, Issue 2, Article 412 (January 8, 2026) [S1]
- **DOI**: https://doi.org/10.3390/s26020412 [S1]
- **Submission/Acceptance**: Received November 7, 2025; Accepted January 5, 2026 [S1]
- **Method**: Dual-channel reverse-distillation network combining RGB and depth modalities with guided feature refinement and cross-modal fusion [S2]

**Reported MVTec 3D-AD Performance:**
- **Image-level I-AUROC**: 97.1% [S2]
- **Pixel-level PRO**: 98.8% [S2]
- **Context**: Paper states these metrics "surpass current state-of-the-art multimodal methods on this benchmark" [S2]

### Top 5 SOTA Methods on MVTec 3D-AD

Ranked by reported mean image-level AUROC/I-AUROC:

| Rank | Method | Mean I-AUROC | Pixel-Level Metric | Publication | arxiv/DOI | Year |
|------|--------|--------------|-------------------|-------------|-----------|------|
| 1 | **DCRDF-Net** | **97.1%** | **PRO: 98.8%** | Sensors 26(2):412 | https://doi.org/10.3390/s26020412 | 2026 |
| 2 | **CPMF** | **95.15%** | **PRO: 92.93%** | Pattern Recognition (2024) | arxiv 2303.13194 | 2023 |
| 3 | **Shape-Guided** | **94.7%** | **AUPRO: 97.6%** | ICML 2023 (PMLR v202:6185–6194) | OpenReview IkSGn9fcPz | 2023 |
| 4 | **M3DM** | **94.5%** | **AUPRO: 96.4%** | CVPR 2023 | arxiv 2303.00601 | 2023 |
| 5 | **AST** | **93.7%** | —not reported— | —not specified in sources— | — | 2022 |

*Alternative ranking by pixel-level PRO (when available):*
- Shape-Guided (97.6% AUPRO) exceeds DCRDF-Net on pixel-level if metric definition differs (AUPRO≠PRO); DCRDF-Net's 98.8% PRO is highest per-region overlap.

### Metric Terminology Notes

- **I-AUROC** (image-level AUROC): defect presence classification per image
- **PRO** (Per-Region Overlap): strict per-region localization metric, integrated to FPR≤0.3
- **AUPRO**: variant or alias; Shape-Guided and M3DM papers report "AUPRO"
- DCRDF-Net explicitly reports **PRO** (98.8%), not AUPRO—highest single-number claim observed
- **Papers with Code leaderboard**: attempted fetch redirected to Hugging Face; no direct public leaderboard URL found, but GitHub repos (M-3LAB/awesome-3d-anomaly-detection) maintain community benchmark lists [S6]

### Method Characteristics

**DCRDF-Net (2026)**
- Multimodal (RGB + depth)
- Key innovation: Perlin-guided pseudo-anomaly generator, guided feature refinement, cross-modal squeeze–excitation gating
- First-published claim on MVTec 3D-AD for this exact method name and configuration

**CPMF (2023)**
- Multimodal (point cloud pseudo-2D + pre-trained 2D networks)
- Ranked #2 best model for Depth Anomaly Detection and Segmentation on MVTEC 3D-AD (Segmentation AUPRO metric) per Papers with Code [S5]

**Shape-Guided (ICML 2023)**
- Geometry-aware (dual memory: shape expert + appearance expert)
- Implicit distance fields for structural anomalies
- Strong pixel-level AUPRO (97.6%), slightly lower image-level than M3DM

**M3DM (CVPR 2023)**
- Multimodal hybrid fusion (point features aligned to 2D, unsupervised feature fusion, multiple memory banks)
- Most cited 2023 baseline; balanced metrics (94.5% I-AUROC, 96.4% AUPRO)

**EasyNet (2023)**
- Reconstruction-based (RGB-D, no pre-trained models required)
- Reported as 92.6% AUROC (specific metric level unclear from sources); lower than top tier but noteworthy for not requiring pre-training

**AST (2022)**
- Earlier method; 93.7% I-AUROC per search results; details limited in collected sources

### Citation & Verification Status

- **DCRDF-Net metrics**: Verified from NCBI/MDPI full article and paper abstract; published peer-reviewed journal [S1, S2]
- **CPMF metrics**: Verified from arxiv abstract and Papers with Code entry [S4, S5]
- **Shape-Guided**: Conference paper (ICML 2023); metrics confirmed via GitHub repo and OpenReview [S7, S8]
- **M3DM**: Conference paper (CVPR 2023); arxiv abstract states outperforms SOTA; specific metrics found in search results referencing benchmark tables [S9]
- **EasyNet, AST**: Lower confidence on pixel-level metrics; image-level AUROC cited from abstracts and search summaries [S3, S10]

## Open Questions

- **Private leaderboard state**: No publicly accessible Papers with Code or official MVTec 3D-AD leaderboard found during search; benchmark tracking relies on individual paper reports
- **Metric standardization**: PRO vs AUPRO definitions may vary slightly across papers; precise alignment requires full paper review
- **Post-2026 methods**: Search limited to 2026-07-14; newer methods may exist on arxiv with later submission dates
- **EasyNet & AST full metrics**: Pixel-level PRO/AUPRO not verified for these methods in this pass; search found only image-level scores

## Sources

- [S1] MDPI Sensors Journal: "DCRDF-Net: A Dual-Channel Reverse-Distillation Fusion Network for 3D Industrial Anomaly Detection" — https://www.mdpi.com/1424-8220/26/2/412 — Volume 26, Issue 2, Article 412, Published January 8, 2026 — accessed 2026-07-14
- [S2] NCBI PMC (full text): DCRDF-Net paper, metrics section — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12846035/ — accessed 2026-07-14
- [S3] EasyNet arxiv abstract — https://arxiv.org/abs/2307.13925 — "EasyNet: An Easy Network for 3D Industrial Anomaly Detection" (arXiv:2307.13925v1) — accessed 2026-07-14
- [S4] CPMF arxiv abstract — https://arxiv.org/abs/2303.13194 — "Complementary Pseudo Multimodal Feature for Point Cloud Anomaly Detection" (arXiv:2303.13194) — accessed 2026-07-14
- [S5] Papers with Code: "Complementary Pseudo Multimodal Feature for Point Cloud Anomaly Detection" — https://paperswithcode.com/paper/complementary-pseudo-multimodal-feature-for — accessed 2026-07-14
- [S6] GitHub: M-3LAB/awesome-3d-anomaly-detection — https://github.com/M-3LAB/awesome-3d-anomaly-detection — accessed 2026-07-14
- [S7] GitHub: jayliu0313/Shape-Guided repository — https://github.com/jayliu0313/Shape-Guided — Shape-Guided Dual-Memory Learning for 3D Anomaly Detection [ICML2023] — accessed 2026-07-14
- [S8] ICML 2023 Proceedings: "Shape-Guided Dual-Memory Learning for 3D Anomaly Detection" — https://proceedings.mlr.press/v202/chu23b.html — Proceedings of Machine Learning Research, Volume 202, pages 6185–6194 — accessed 2026-07-14
- [S9] M3DM arxiv: https://arxiv.org/abs/2303.00601 — "Multimodal Industrial Anomaly Detection via Hybrid Fusion" (CVPR 2023) — accessed 2026-07-14
- [S10] Web search result: "AST achieved 93.7% image-level AUROC on MVTec 3D-AD" — search results snapshot — accessed 2026-07-14
