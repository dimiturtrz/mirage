# Field map & coverage — what to learn for 3D surface-anomaly perception, and how much

A **yardstick**: which fields this project sits on, which *view* of each we need, and a checklist to
answer "have we covered enough?" without guessing. 3D anomaly detection is an intersection — take each
field only to the depth that explains *our data and our claims*.

## Where this knowledge lives (the intersection)
| field | who owns it | what it explains for us | our depth |
|---|---|---|---|
| **3D / computational geometry** | graphics / geometry | point clouds, surfaces, **normals**, curvature, local descriptors (**FPFH**) | working fluency — our hands for the geometry |
| **3D sensing (structured light)** | metrology / optics | how the Zivid scanner makes a position map; **sensor noise** (flying pixels, grazing-angle) | *enough to explain the data + its failure modes* |
| **Computer vision / deep features** | CV / ML | CNN backbones, **frozen pretrained features**, feature space, patch features | home turf (carries from prior work) |
| **Anomaly detection theory** | ML | train-on-normal, the paradigms (**reconstruction · memory-bank · synthesis-discriminative**) | core — the contribution's subject |
| **Statistics / eval** | stats | ROC/PRC, **AU-PRO**, calibration (ECE), the like-for-like comparison | working fluency — the rigor layer |
| **Synthesis / sim-to-real** | graphics + ML | defect generation, the domain gap, digital-twin reconstruction (single-view) | the Stage-1 contribution |
| **Linear algebra / ML foundations** | maths | PCA (normals!), nearest-neighbour, distances | working fluency, not proofs |

The lane is **3D industrial inspection** = geometry + 3D sensing (what the data is) × deep features + anomaly
theory (how to detect) × stats (how to judge rigorously) × synthesis (the sim-to-real engine). The
deliverable lives in the detection + robust-eval + synthesis layers; take geometry/sensing to the depth
that makes the data and its failure modes legible.

## Target competency (the bar) — three layers
**F) Foundations** (under everything): PCA / eigenvectors (→ surface normals) · nearest-neighbour + distance
metrics (→ memory bank) · ROC/PRC + calibration. Working fluency, not proofs.

**G) Geometry-enough**: what a position map / point cloud / 2.5D scan *is*; surface normals + curvature;
FPFH as a handcrafted geometric descriptor; the structured-light sensor's noise. *Enough to explain the
data and why the geometry is hard — not to write a renderer.*

**D) Detection stack** (the real work): the three anomaly paradigms and **why feature-space beats pixel
space** · PatchCore (memory bank) · multimodal RGB+3D fusion (the SOTA) · **robust evaluation** (AU-PRO,
per-defect diagnostics, calibration) · the sim-to-real engine (synthesis, the gap, the twin). This is the
contribution.

## Coverage checklist (✅ theory done · ⬜ pending)
- [x] FPFH, surface normals, neighborhoods — handcrafted geometric descriptor (`2026-06-25`)
- [x] DRAEM synthesis: realism-vs-signal, multi-seed discipline (`interpretations/synthesis/2026-07-06_draem-synthesis-comparison`)
- [ ] **the anomaly paradigms + why feature-space beats pixel reconstruction** ← THE headline; next
- [ ] PatchCore mechanism (frozen features, coreset, NN-distance)
- [ ] 3D data representation (position map, point cloud, 2.5D, background/valid mask)
- [x] structured-light sensing + its noise — triangulation, z² law, 5 noise modes (`2026-07-06_structured-light-sensing-and-noise`, quiz 52%)
- [ ] AU-PRO — what it measures, why per-region, the FPR-0.3 integration
- [ ] multimodal RGB+3D fusion — M3DM / current SOTA (how fusion buys the last ~8pp)
- [ ] sim-to-real gap + single-view reconstruction (the digital twin)

## Academic mapping (which university courses this is)
mirage ≈ a targeted vertical of a **Master's in 3D Computer Vision** — the geometry + 3D-deep-learning +
industrial-inspection slice. It collapses to ~four courses:

| # topic | university subject | canonical course / text |
|---|---|---|
| Foundations | Linear Algebra · Prob/Stats | Strang; Wasserman *All of Statistics* |
| #1 3D data representation | **3D Computer Vision** | Stanford **CS231A**; Szeliski *Computer Vision* |
| #2 structured-light sensing + noise | 3D imaging / range sensing | CS231A; Szeliski §12–13 |
| #3 normals · curvature · FPFH | **Geometry Processing** | Stanford **CS468**; Rusu *FPFH* (2009) |
| #4 feature-space vs reconstruction | **Deep Learning for CV** | Stanford **CS231n**; Bishop *PRML* |
| #5 PatchCore | Advanced CV / Visual Recognition | PatchCore (Roth 2022) |
| #6 AU-PRO / calibration | ML evaluation / Trustworthy ML | (inside ML courses) |
| #7 multimodal RGB+3D fusion | **3D Deep Learning** | CS231n/CS468 3D-DL; M3DM (2023) |
| #8 sim-to-real + reconstruction | **Multi-View Geometry** · Computer Graphics | **Hartley & Zisserman**; Marschner–Shirley |

Four buckets: **CS231n** (deep features/anomaly ML → #4,5,7) · **CS231A/Szeliski** (3D vision → #1,2,8) ·
**CS468** (geometry → #3) · a **stats/eval** thread (#6 + rigor). Depth = "explains our data and
claims," not exam depth.

## The circuit (see learning/README)
research (teacher grounds → `research/`) → theory writeup here → **quiz** (on demand) → log answers →
sharpen the plan. Domain facts source-grounded; wrong answers stay in the log — the ramp is real.
