# Research Notice Board

## Open Questions

| Question | Why it matters | Owed by |
|----------|----------------|---------|
| The ⚠️ markers in `docs/PLAN.md` "SOTA / OSS landscape" (BTF/PatchCore-3D exact numbers, AV DA %s) | A dedicated research pass owed before any land in README | Stage 0→1 |

## Settled Findings

| Topic | Deep-Dive | TL;DR |
|-------|-----------|-------|
| M3DM SOTA number on MVTec 3D-AD | `2026-06-29_3d-anomaly-sota-and-sim2real-benchmark.md` | "~0.96" = **0.964 pixel AU-PRO** (3D+RGB), CVPR 2023 — README correct (earlier 0.945 worry was image-AUROC conflation). |
| Sim-to-real 3D-anomaly benchmark | `2026-06-29_3d-anomaly-sota-and-sim2real-benchmark.md` | **SiM3D** (arXiv 2506.21549, 2025) IS the first synth→real 3D-anomaly benchmark — "no benchmark exists" claim was false; README + PLAN fixed to cite it + reframe contribution. |
| Current MVTec 3D-AD SOTA (landscape) | `2026-07-06_mvtec3d-sota.md` | SOTA moved past M3DM: current top **DCRDF-Net 0.971/0.988** (2026); cluster MMRD/Shape-Guided ~0.95/0.976. M3DM (0.945/0.964) now mid-pack. Docs' stale "SOTA=M3DM" updated. Our gap ~8pp AU-PRO / ~15pp img (rgb-only vs multimodal fusion). |

## Progress Log

- **2026-06-29**: Pre-public claim-verification pass. M3DM 0.96 stands; "no sim-to-real benchmark"
  corrected (SiM3D exists) in README + PLAN. Stage-0 results remain measured (not from literature).
- **2026-07-06**: SOTA-verify pass. Current top = DCRDF-Net 0.971/0.988 (verified from primary);
  updated RESULTS.json/md + README from stale M3DM. TransFusion 0.992 is MVTec AD *2D*, not conflated.
