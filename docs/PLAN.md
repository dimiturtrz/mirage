# mirage — build plan

> **Personal learning project — a physics-based synthetic-data engine for 3D perception: generate defect scenes, detect anomalies. Edge-deployable.**

Working name `mirage` (synthetic that must survive contact with the real). Folder/repo may differ; rename freely.

Sibling project — same build philosophy, different domain: **[systole](../cardiac-seg/)** (cardiac MRI segmentation → ejection fraction → honest cross-scanner eval). mirage is the same *engine-first, honest-eval* pattern, applied to 3D inspection. The structural and honesty conventions below are lifted from it deliberately.

---

## The center (what this actually is)
**Not** a "3D perception stack." It's a **synthetic-data generation engine** for perception, applied to a new 3D modality. The engine is the through-line:

**data discovery → synthetic scene generation → (closed-loop curriculum) → train/score → feed back → honest sim-to-real eval.**

That loop is carried over from prior acoustic-detection work (physics-based multichannel scene synthesis + closed-loop curriculum + calibration/eval rigor); the **3D modality is the deliberate stretch** — the same move systole made (leverage the data-engine + eval strength, learn the new field).

- **Engine = the leverage** (signature strength). Physics/sim-grounded data generation; the realer the sim, the better the data.
- **Anomaly = the task** (close to existing strength — unsupervised, train-on-normal → score-deviations).
- **Honest sim-to-real eval + calibration = the contribution** (the field lacks it; it's the recurring strength).
- **3D point cloud = the new domain** (the only genuinely new skill; treat as a ramp).

## What we're building (one line)
A physics-based synthetic-defect generator for 3D inspection, an anomaly detector trained on it, and the honest measurement of how much that detector loses on real scans it never trained on — edge-deployable.

---

## The spine: one honest, bounded capability
systole's power is one through-line — *segment the heart → measure EF → show where it fails* — with the **honest-hard-part as the contribution** (does EF survive a scanner change?), not a leaderboard number. mirage mirrors this:

**generate synthetic defects → detect anomalies → quantify the sim-to-real gap honestly → show where it fails.**

### Headline numbers — decided up front (systole's lesson)
systole hits because the numbers are concrete and tied to a bar: *EF MAE 5.9%, bias −5.2%, LoA ±13, against a ±5% clinical bar.* mirage commits to its equivalents on day one:
- **AU-PRO / AUROC** on MVTec 3D-AD held-out (real-trained ceiling) — the standard benchmark *is* the SOTA baseline to quarantine against, the role nnU-Net played in systole.
- **Sim-to-real gap** in percentage points (synth-only vs real-trained) — the headline honest number.
- **Calibration** (ECE / reliability) under domain shift — the rare-class + calibration signature.
Per-condition diagnostics (defect type, material, lighting) stratify the failure, the way systole stratified EF error by pathology/vendor. (Calibration + ECE + domain-shift detection are already in the toolbox — carried, not learned.)

### The differentiator: closed-loop curriculum (carried, novel in 3D)
The acoustic engine adapts generation difficulty per epoch from the trainer's previous-epoch score (per-class SNR shift), coupling generator to trainer. Almost nobody does this in 3D sim-to-real. Port it: defect difficulty / rarity / rendering realism adapts to the detector's current weakness. A genuine contribution, not a reimplementation.

---

## The ramp (real spine first, then the engine)
1. **Stage 0 — Static, real data.** Start on **MVTec 3D-AD** (real industrial 3D anomaly). Stand up the **reconstruction-VAE** anomaly model (rebuild normal → residual + latent distance = anomaly) + the **eval harness** (rare-class metrics, calibration, per-condition diagnostics) on real data. Prove the spine before synthetic. **First commit = MVTec 3D-AD + eval harness.** = systole's Gate 1 (presentable) = **the public-flip trigger.**
2. **Stage 1 — Synthetic engine + sim-to-real (the differentiator, the center).** Render synthetic defects (Replicator/Isaac, or Blender/BlenderProc). Train synth → test real; **quantify the gap honestly.** Reproduce ONE known DA method and *measure* it — don't try to beat it. Then layer the **closed-loop curriculum** on top.
3. **Stage 2 — Edge deploy.** ONNX → TensorRT + FPS/latency/memory benchmark (Jetson if available, else CPU/GPU; RPi/edge-accel experience carries). Target <1% AUROC/AU-PRO loss vs the full-precision model.
4. **After the spine — two expansion axes** (perception breadth · engine depth). Optional, sequenced *after* the public flip — not pulled forward. See **Expansion axes** below.

## Comparisons (the triad IS the contribution)
Like systole's nnU-Net baseline, report three honestly:
1. Real-trained ceiling (train real → test real)
2. Synthetic-only (train synth → test real) = the gap
3. Synthetic + DA (+ closed-loop curriculum) = gap closure

That triad + calibration + per-condition diagnostics is the defensible result — not a leaderboard number.

---

## Expansion axes (after the Stage 0–2 spine)
Two axes branch off the spine. Both optional, both sequenced *after* the public flip — mapped here so the ladder isn't lost, not to be pulled forward.

**Axis 1 — perception breadth** (more role coverage)
- **Segmentation** — dense spatial labeling ("which part is defective"). The *least-him* task (pure new skill), but the lever that completes the CV-platform role; shares the eval harness, so cheap. **Keep reachable.**
- **Detection + tracking** — 2D experience exists; 3D is a stretch and low role-ROI (opens mostly gated roles). Deprioritized.
- **Multi-modal fusion** — RGB + geometry (+ thermal) join via the calib + timestamp pattern.

**Axis 2 — engine depth** (the carried R&D ambition; toward the simulation-first stack)
- **The hinge:** building the **Stage 1 generator in Isaac/Omniverse (not Blender)** plants this axis — same Stage-1 deliverable, but in the stack that continues here.
- **T2 — temporal / scenario simulation:** deepen the static defect-gen into **video-shape** sequential sim (a scene over time) in Isaac Sim / Omniverse / USD. Buys the USD/Omniverse/Isaac-Sim + dataset-gen-as-temporal competence (the full multimodal scene-simulation ambition: light + geometry, later + audio). VLA models are video/sequence-conditioned, so this is their native input.

**Charter boundary — where mirage stops.**
- **T2 stays in mirage** — it's *generating data*, which is the engine charter.
- **T3 (train a VLA + RL on that data) is OUT** — that's control/action = door-2, forbidden by the restraint rule (no control, manipulation, or RL). T3 is a separate project/limb. **mirage ends at "generate the data + perceive"; training action models on it is the next project, not a mirage stage.**

---

## Structure — three pieces, general → specific (systole's shape)
systole's memorability is the two visualizers that make it demonstrable: `mri-sim` (understand the data the model consumes) + `cardioview` (in-browser ONNX, beating 3D heart on held-out cases). mirage clones the three-piece shape. **A piece is added only when it's real — no empty speculative folders.**

| systole | mirage | role |
|---|---|---|
| `mri-sim` — MRI signal-pipeline 3D viz | **synth-gen viz** — Replicator/Blender render-pipeline preview | *understand the data the model consumes* — here, how synthetic defects are generated. This is the engine (the center) made visible; surfaces USD/Omniverse literacy as a side effect. |
| `cardioview` — in-browser ONNX, beating 3D heart | **pointcloud viewer** — three.js / Open3D-web | *see the model work* — predicted anomaly heatmap on the 3D point cloud; drop-your-own-scan, in-browser ONNX inference. |
| `cardioseg` — data → model → measure → eval | **pipeline** — data → model → anomaly → eval harness | *the science layer* — the eval harness is the contribution. |

### learning/ track (systole's "how I ramp into the field")
Mirror systole's `learning/<date>_<topic>.md` + glossary + on-demand self-quizzes, and the circuit: **research (teacher grounds it → `research/`) → theory writeup (`learning/`) → quiz → sharpen the plan.** The 3D-geometry/point-cloud theory is where the ramp effort goes. Build log = git history; theory artifacts = `learning/`.

### Reuse from systole + the acoustic engine wholesale (don't reinvent)
- **`paths.yaml` one-root config** + data-out-of-repo + per-source adapters → a common schema. MVTec (real) + synthetic renders = a 2-source cloud, same adapter pattern.
- **Calibration + eval harness** — ROC/PRC-AUC, Brier, ECE, KS-test domain-gap, distribution-shift detection — carried from acoustic, retargeted to 3D anomaly.
- **Closed-loop curriculum** — the generator↔trainer coupling, ported.
- **Model cards + auto-ONNX-export at train-end** (per-run, parity-gated INT8).
- **Cross-implementation parity tests** — the train-vs-deploy preprocessing-skew discipline (Python ↔ Rust ↔ FPGA bit-true) applies directly to ONNX/TensorRT export parity.
- **README skeleton:** one-line → the honest question → three pieces → the number that matters first → stratified where-it-fails → honest limits → data → quickstart → tests → how-it's-built → license.
- **Test layout:** unit (equivalence-class) + integration (module-pairs).

---

## Input data & handling (MVTec 3D-AD)
**Format (verified on-disk):** per sample, an 800×800 grid with three synced channels — `rgb/NNN.png` (color), `xyz/NNN.tiff` (per-pixel x,y,z position map, meters, from a Zivid structured-light scanner), `gt/NNN.png` (defect mask, test only). Single-view 2.5D surface (one scan/sample). `train`/`validation` = good only; `test` = good + defect types with masks.

**Representation choice = keep the organized grid (position map as a 3-channel image), not unordered points.** Why: the reconstruction VAE wants a fixed-size tensor (the grid is one); it plays the 2D-CNN strength (depth-image-shaped); and **per-pixel reconstruction error is a pixel-level anomaly map for free — exactly the AU-PRO localization target.**

**Preprocess contract (raw → `processed/mvtec3d/<cat>/<split>/`):**
1. Load xyz + rgb (+ gt for test).
2. **Background mask** — `xyz != (0,0,0)` (the only background marker; rgb/gt don't encode it; verified exactly-zero, no NaN).
3. **Flying-pixel removal** — statistical outlier removal (structured-light artifacts at depth edges are non-zero, so they survive the background mask). Update mask.
4. **Crop** to object bbox, **resize** to fixed (e.g. 256²).
5. **Normalize** — geometry to a *known physical* scale (center + fixed scale; xyz already in meters), NOT dataset statistics (the carried calibrate-to-physical-units principle). RGB /255.
6. **Store** the normalized position-map tensor + valid mask + (test) gt + meta.

**Two carried-from-acoustic specifics:**
- **Masked reconstruction loss** — compute loss only on valid (object) pixels; don't spend VAE capacity rebuilding background.
- **Anomaly scoring** — per-pixel residual → pixel map (AU-PRO); top-residual aggregate → image AUROC; **+ latent Mahalanobis distance** (the VAE-latent trick); + ECE calibration.

**Channels:** start geometry-only (xyz, 3ch); later add rgb (→6ch) for a **geometry-vs-rgb-vs-fused mini-triad** (fusion strength + an honest comparison).

## Tech stack
- **Data:** MVTec 3D-AD (real) → synthetic defects (Replicator/Isaac, or Blender/BlenderProc)
- **Processing:** Open3D (PCL if C++ needed) — *new; the ramp target*
- **Models:** **reconstruction VAE — built, not borrowed** (the one justified exception to use-don't-reimplement: it's the contribution, ported from acoustic anomaly work). Libraries (Open3D, training infra) for plumbing only. Memory-bank methods (PatchCore/M3DM) as comparison baselines, used not rebuilt.
- **Deploy:** ONNX → TensorRT (→ Jetson); RPi/edge-accel experience carries
- **Viewers:** three.js / vtk.js / Open3D-web (in-browser ONNX), TypeScript
- **Eval:** custom harness — rare-class metrics, calibration-under-shift, sim-to-real gap, per-condition diagnostics ← **the contribution**

### Stack landscape (where things sit)
- **Formats:** USD (3D scenes), ONNX (models)
- **Sim / data-gen:** Omniverse (platform) · Isaac Sim (robotics sim) · **Replicator (synthetic data-gen — the center)** · CARLA (driving) · Blender/BlenderProc (generic). **Isaac Lab = RL/control = skip.**
- **Datasets:** anomaly (MVTec 3D-AD — **core**) · seg/AV only on the breadth axis (ScanNet/S3DIS indoor; KITTI/nuScenes AV) · synthetic (our renders — core; KITTI-CARLA/SynLiDAR/3D-FRONT as references)
- **Processing:** Open3D, PCL
- **Models:** anomaly — reconstruction VAE (ours, built) · M3DM/BTF/PatchCore-3D (baselines, used). Detection/seg libs (MMDetection3D, OpenPCDet, Open3D-ML, PyTorch3D) only on the breadth axis.
- **Deploy:** ONNX → TensorRT → Jetson

Generating with Replicator/Isaac Sim (not Isaac Lab) buys USD/Omniverse/Isaac literacy via the *data-gen* half — no RL. It's also the visual instantiation of the carried R&D ambition (multimodal scene simulation as a training-data generator).

---

## SOTA / OSS landscape (⚠️ = verify before it lands in README; a dedicated research pass is still owed here)
**Core path — 3D anomaly detection on MVTec 3D-AD:**
- **Memory-bank / feature methods** (the SOTA baselines to quarantine against, used not rebuilt): BTF (handcrafted FPFH + RGB) ⚠️, M3DM (multimodal RGB + point fusion) ⚠️, PatchCore-3D ⚠️. Top methods report image-AUROC ≈0.9+ and pixel-AU-PRO ≈0.9+ ⚠️.
- **Reconstruction-based** (our family): autoencoder / dual-branch RGB-D reconstruction (e.g. arXiv 2311.06797) ⚠️. Typically *lags* memory-bank on the leaderboard ⚠️ — expected, and fine: the contribution is the engine + eval rigor, not the number (the systole nnU-Net-baseline move).
- **Libs:** Anomalib (2D-centric, limited 3D) ⚠️; method repos (M3DM, BTF) used directly.
- **White space (the opening):** no standardized **sim-to-real** benchmark for 3D anomaly; synthetic-defect generation + honest gap measurement is largely unexplored — the differentiator. Quantization + domain-shift compound effect unmeasured (directly in the toolbox: distribution-shift detection + edge quantization).

**Breadth / AV axis only (Axis 1, or a future pivot — NOT the core anomaly path):**
- Detection/seg libs: MMDetection3D (6.5k★), OpenPCDet (5.6k★) ⚠️.
- AV sim-to-real DA (cross-domain reference): ePointDA, STAL3D/UADA3D, Synth-it-Like-KITTI; ~5–17% AP3D CARLA→KITTI ⚠️.
- Synthetic AV: CARLA, Isaac Sim; KITTI-CARLA, SynLiDAR ⚠️.

**Edge:** PyTorch → ONNX → TensorRT; target sub-1% metric (AUROC/AU-PRO) loss under quantization ⚠️.

## SOTA stance / honesty rules
1. **Don't chase AUROC/AU-PRO SOTA.** Use SOTA libs, reproduce a SOTA-ish DA method, **contribute the eval rigor + the data engine** the field lacks.
2. **Differentiator = the engine + the eval, not the model.** The detector is commodity; say so.
3. **Cite prior art** (MVTec 3D-AD, M3DM/BTF, reconstruction-AD methods; CARLA/Isaac/MMDet3D only where the breadth/sim axes touch them) — informed, not naive.
4. **Verify every ⚠️** against a primary source before it lands in the README.
5. Thesis framing: *"a synthetic-data engine + the honest sim-to-real evaluation the field lacks,"* not *"I built synthetic→real detection."*
6. **Honest-limits section, measured not assumed** (systole's "clinical-grade gap"): claim edge deploy (real), never production-scale serving; 3D is a ramp — say so; state the gaps as numbers.
7. Build private → **flip public at Stage 0** (MVTec + eval harness presentable, prior art cited); steady real commits after.

---

## Start here
**MVTec 3D-AD + the eval harness, static, real data.** First commit. The engine (Stage 1) builds on top. That commit is the public-flip trigger.
