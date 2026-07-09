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
| M3DM integration plan (synthscape-iwq) | `2026-07-09_m3dm-integration.md` | Repo `nomewang/M3DM` MIT @commit `62b7f468`, MVTec-3D-specific. **Top risk: `pointnet2_ops`+`KNN_CUDA` pinned torch1.9/CU11.3 won't build for sm_120** (needs source patch + `TORCH_CUDA_ARCH_LIST=12.0` + nvcc `-O2`). Seam = `Features(nn.Module)` w/ abstract add-to-bank/predict → maps to our `AnomalyMethod`. Full fusion = multi-day; **lighter: Point-MAE feats→our bank = ~weekend**, most of the learned-geometry gain w/o the fusion machinery. Inference 6.5 GB/0.51 FPS (fine on 5090); wants raw MVTec-3D not our s256. |
| Edge accelerator op-support (deploy) | `2026-07-08_edge_accelerator_landscape.md` | Coral/Hailo/RKNN are int8 fixed-graph conv accelerators: **PatchCore kNN-vs-stored-bank tail is inherently host-CPU** (top-k/cdist/fp32-bank not accelerated); pure-conv int8 AE maps cleanly onto all. Only Jetson/TensorRT (full ONNX, native TopK/Gather) runs both on-device. |
| PatchCore bank on NPU — reducible? | `2026-07-08_patchcore_bank_on_npu.md` | **Matmul half reducible, argmin/top-k half not** on Coral/Hailo. Distance-vs-bank = frozen Conv1x1 (W=−2B, bias=‖b‖²) IS accelerator-native; but selection (top-k absent Coral/Hailo, RKNN TopK unsupported/ArgMin ok→k=1 maybe) truncates to host. Int8+PQ bank fits (334 KB @10k×256) but costs ~9 pp img-AUROC (0.95→0.86). **No prior art ran the lookup on a fixed-function NPU** — all edge PatchCore = host-CPU or Jetson-GPU. Field's on-NPU answer: bank-free conv model (EfficientAD/AE). |

## Progress Log

- **2026-06-29**: Pre-public claim-verification pass. M3DM 0.96 stands; "no sim-to-real benchmark"
  corrected (SiM3D exists) in README + PLAN. Stage-0 results remain measured (not from literature).
- **2026-07-06**: SOTA-verify pass. Current top = DCRDF-Net 0.971/0.988 (verified from primary);
  updated RESULTS.json/md + README from stale M3DM. TransFusion 0.992 is MVTec AD *2D*, not conflated.
- **2026-07-08**: Edge-deploy research. Landscape (op-support) + PatchCore-bank-on-NPU follow-up.
  Bank lookup reducible to backbone+distance-matmul on-NPU, but argmin/top-k stays host on Coral/Hailo;
  no prior art puts it on a fixed-function NPU. Closes the "reduced-bank on NPU?" open Q from landscape.
