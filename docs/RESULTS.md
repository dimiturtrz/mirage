# Results — anomaly detection on MVTec 3D-AD (Stage 0 detector + Stage 1 sim-to-real)

Measured first-stage findings. **Per-category models** (the MVTec convention), scored through our own
eval harness — image-level **AUROC** (detection) + pixel-level **AU-PRO** (localization). These are
*our* runs through *one* eval harness, **not** the papers' leaderboard numbers.

## Headline: reconstruction fails, feature-memory-bank works
<!-- results:methods -->
| method | modality | mean img-AUROC | mean pixel AU-PRO | params (M) | GFLOPs | int8 (MB) |
|---|---|---|---|---|---|---|
| Reconstruction (VAE / +dropout / inpaint / fused) — ours | xyz (+rgb) | 0.486 | 0.095 | 30.8 | 2.3 | 30.8 |
| DRAEM (synthesis-discriminative, crude Perlin) · bagel — ours | xyz | 0.610 | 0.360 | 15.5 | 48.4 | 15.5 |
|   ↳ realistic-synth v3 (normal-displaced, on DRAEM) · bagel — ours | xyz | 0.790 | 0.320 | 15.5 | 48.4 | 15.5 |
| BTF (FPFH memory bank) — ours | geometry | 0.674 [0.641, 0.707] | 0.650 [0.637, 0.662] | — | — | — |
| **PatchCore (feature memory bank)** — ours | rgb | 0.838 [0.812, 0.866] | 0.902 [0.892, 0.910] | 24.9 | 24.0 | 24.9 |
| Feature-recon / RD4AD-lite — ours | rgb | 0.805 [0.777, 0.833] | 0.907 [0.899, 0.915] | 6.2 | 13.9 | 6.2 |
| Fused (PatchCore-rgb + BTF) — ours | rgb + 3D | 0.840 [0.813, 0.868] | 0.929 [0.923, 0.935] | 24.9 | 24.0 | 24.9 |
| SOTA (DCRDF-Net)† | rgb + 3D | ~0.97 | ~0.99 | — | — | — |
<!-- /results:methods -->

<sub>**Cost columns** (`params / GFLOPs / int8`) are the **neural forward at 256²** measured with torch's
FlopCounterMode (`python -m surfscan.deploy profile`) — backbones counted truncated to the read layer
(layer4+fc pruned). They surface the deploy tradeoff the accuracy columns hide: **feature-recon matches
PatchCore's AU-PRO (0.907 vs 0.902) at a quarter the params (6.2M vs 24.9M) and no memory bank.**
PatchCore/fused additionally carry a **20k×1536 feature bank** (117 MiB fp32 / 1.65 MiB int8+PQ) whose
kNN tail is host-side on every commodity NPU — the bank-memory model and the per-accelerator projection
live in the single `docs/deployment.json` (`surfscan.deploy bank` / `project`). The
deploy-driven detector choice is read off these numbers in
[`interpretations/deploy/2026-07-15_deploy-cost-model.md`](../interpretations/deploy/2026-07-15_deploy-cost-model.md).</sub>

<sub>¹ fused all-10 (<!--r:fused.au_pro-->0.929<!--/r-->) is the **top detector** — **+1.5pt** over rgb-only PatchCore (<!--r:patchcore.au_pro-->0.902<!--/r-->, same 0.1-greedy coreset), CIs near-disjoint ([0.911, 0.923] vs [0.892, 0.910]). Geometry fusion adds an orthogonal signal: biggest gains on geometry-rich categories (carrot 0.991, dowel 0.977, potato 0.979), small dips only where our preprocessing-limited BTF is weakest (cookie 0.811, foam 0.835). Fusion **pays even with degraded FPFH** — native-res geometry is further headroom. Single representative run (multi-seed to harden). Deployable trade-off: rgb-only stays the simplest edge path (no geometry pipeline); fused leads on accuracy.</sub>

**The lesson:** two *independent* feature methods — a memory bank (PatchCore) and a
reconstruction AE (feature-recon) — both land at **AU-PRO ~0.90**, vs pixel-reconstruction <!--r:recon.au_pro-->0.095<!--/r-->.
So what matters isn't memory-bank-vs-reconstruction; it's the **feature space**.
Reconstruction was never the wrong idea — *raw-pixel space* was (normal geometric complexity
dominates the residual). Move the same reconstruction into a frozen-backbone feature space (where
normal complexity is already normalized) and it localizes fine. That's the Stage-0 finding.

² **DRAEM** (synthesis-discriminative), bagel xyz, **3-seed** (noise discipline): crude Perlin
**0.36 ± 0.09** au_pro — the earlier single-seed **0.48 was a lucky draw** (a robustness fix to our own
number). My first "realistic" synth (v2) scored **lower, 0.18 ± 0.02** — but that was **broken geometry,
not a law that realism hurts DRAEM:** v2 displaced defects along the **world-z axis, not the surface
normal**, so on the bagel's curved sides a "dent" was a lateral shear. **Fixed (v3, normal-displaced;
`core/geometry.py`):** re-running the same 3-seed sweep, realistic-v3 = **0.32 ± 0.07 au_pro (tied with
Perlin, the deficit gone)** and **0.79 img_auroc vs Perlin 0.61 (higher on all 3 seeds, decisively on 2 —
seed-1 a near dead-heat)**. So realistic-normal is now *tied on localization, better on image-detection*. The eval spine refuted the assumption, exposed
a cherry-picked number, then caught the over-correction. See
`interpretations/synthesis/2026-07-06_draem-synthesis-comparison.md`.

### Per-defect-type (PatchCore-rgb, pooled across categories)
Most defect types localize 0.87–0.96 AU-PRO; weak spots: **thread 0.59**, color hardest to *detect*
(image-AUROC 0.59). The stratified failure (like systole's EF-by-pathology).

**<!--r:recon.au_pro-->0.095<!--/r--> → <!--r:patchcore.au_pro-->0.902<!--/r-->.** Reconstruction-residual is ≈ random at localization. The reason — *measured*
across four reconstruction variants (vanilla VAE, +dropout, masked-inpainting, rgb-fused), all
AU-PRO ≈ 0.014–0.095:

> Raw-residual tracks geometric **complexity** (the bagel rim + structured-light grazing-noise are
> hard to rebuild), **not defect-ness** (a smooth defect rebuilds *fine* → low residual). On bagel
> the defect region's residual is ~9× *lower* than normal regions.

"Compare to normal" (a memory bank of normal features) fixes it: a normal-but-complex region (the
rim) matches a normal rim in the bank → not flagged. That's why the SOTA is memory-bank / feature-
based, not reconstruction — here, measured.

## PatchCore-rgb, per category (the working method)
<!-- results:per_category -->
| category | img-AUROC | AU-PRO |
|---|---|---|
| bagel | 0.938 | 0.900 |
| cable_gland | 0.838 | 0.790 |
| carrot | 0.946 | 0.982 |
| cookie | 0.813 | 0.877 |
| dowel | 0.916 | 0.970 |
| peach | 0.883 | 0.935 |
| potato | 0.860 | 0.969 |
| rope | 0.908 | 0.974 |
| tire | 0.650 | 0.920 |
| foam | 0.633 | 0.700 |
| **MEAN** | **0.839** | **0.902** |
<!-- /results:per_category -->

## Stage 1 — sim-to-real triad (the measured gap number)
The Stage-1 deliverable: *"I modeled it" never stands in for "I measured it transferred."* ONE supervised
defect-segmentation U-Net, trained from three label sources, scored on ONE shared real eval half (all-10,
rgb, early-stopped, **ONE deterministic run** with a **bootstrap CI** over the shared eval set — power
from the test-set N, not a cross-seed std). Only the label source varies → the drop IS the sim-to-real gap.
The GAP and DA-CLOSURE deltas carry a **paired** bootstrap CI (same resampled eval images hit both arms, so
shared noise cancels). Full method +
diagnosis → [`interpretations/sim-to-real/2026-07-08_sim-to-real-triad.md`](../interpretations/sim-to-real/2026-07-08_sim-to-real-triad.md).

<!-- results:triad -->
| arm | au_pro [95% boot CI] | img-AUROC | ECE |
|---|---|---|---|
| real → real (ceiling) | 0.794 [0.772, 0.813] | 0.869 | 0.010 |
| synth → real (the gap) | 0.628 [0.604, 0.650] | 0.618 | 0.075 |
| synth+DA → real (AdaBN) | 0.643 [0.618, 0.669] | 0.526 | 0.076 |
<!-- /results:triad -->

- **Sim-to-real gap = <!--r:triad.gap-->0.166<!--/r--> AU-PRO [<!--r:triad.gap_lo-->0.141<!--/r-->, <!--r:triad.gap_hi-->0.190<!--/r-->]** (real <!--r:triad.real.au_pro-->0.794<!--/r--> → synth <!--r:triad.synth.au_pro-->0.628<!--/r-->, paired macro bootstrap — CI
  clears 0, a real gap). One deterministic early-stopped run; power is the test-set N, not seeds.
- **The lever is "don't overtrain on synthetic," not domain adaptation.** Early-stopping the synth arm on a
  held-out *synth* val lifts it 0.537 → **<!--r:triad.synth.au_pro-->0.628<!--/r-->** vs the earlier over-trained run — 100 epochs memorized
  synthetic artifacts that don't transfer; stopping at the synth-val plateau generalizes to real. This
  narrows the gap at the source (hypothesis: the confound is train-length + val-split, measured once).
- **AdaBN is measured-marginal here — a reported reversal.** On the already-strong early-stopped baseline it
  adds **+<!--r:triad.da_closure-->0.015<!--/r--> AU-PRO [−0.000, 0.030]** (CI touches 0 → *not* significant) and does **not** restore
  calibration (ECE <!--r:triad.synth.ece-->0.075<!--/r--> → <!--r:triad.da.ece-->0.076<!--/r-->). The earlier "closes ~46% + restores calibration" was an artifact of the
  *over-trained* baseline (0.537, ECE 0.037) leaving lots of headroom; fix the baseline and the headroom —
  and AdaBN's apparent value — mostly vanishes. Honest negative, kept.
- **Real ceiling 0.79 < PatchCore 0.90 (unsupervised):** supervised-from-scarce-labels loses to the memory
  bank. Synthetic data's payoff here is *volume* (real labels are scarce), not beating them.
- **Curriculum arm not re-measured** under the early-stopped protocol (the earlier 3-seed run scored a
  *negative* 0.487, −0.05 vs uniform, ECE 0.083 — kind-chasing starves batch diversity; fix = adapt
  *difficulty*, not *kind*). Dropped from the fresh bracketed table rather than mix protocols.

## Stage 2 — digital twin vs classical synth (the shape-source number)
Same U-Net, same shared real eval half — but the label **source** is now the **reconstructed** Isaac-rendered
twin (per-category mean-surface mesh from the 244 good scans → path-traced headless → back-projected xyz,
MVTec layout). Two twin arms isolate the two swaps: **twin_grid** = the *same* grid defect on the
reconstructed surface (isolates the **shape** source, defect held constant); **twin_phys** = a *physical*
mesh-deform defect path-traced (isolates the **defect model**). Run on **both channels** — the twin's
contribution is geometry (xyz), not appearance (rgb). Full method + diagnosis →
[`interpretations/twin/2026-07-15_twin-vs-classical.md`](../interpretations/twin/2026-07-15_twin-vs-classical.md).

<!-- results:twin -->
| arm (label source) | rgb AU-PRO | xyz AU-PRO |
|---|---|---|
| real → real (ceiling) | 0.794 | 0.726 |
| synth (classical) → real | 0.628 | 0.565 |
| twin_grid → real | 0.300 | 0.373 |
| twin_phys → real | 0.081 | 0.321 |
<!-- /results:twin -->

- **xyz is the twin's axis — and the data proves it.** Every twin arm jumps rgb→xyz; twin_phys goes from
  **0.081 (≈ random)** to **0.321 (real localization signal)**. A physical mesh-dent is near-invisible in the
  twin's uniform appearance but shows up in geometry. A physically-valid method regressing on rgb meant the
  *rgb test lacks the twin's axis*, not that the twin is bad — scored on xyz, the twin transfers.
- **But the twin does not beat classical synth — even on xyz.** shape-source (twin_grid − synth, defect held
  constant) is **negative on both channels: <!--r:twin.shape_source_rgb-->-0.327<!--/r--> rgb,
  <!--r:twin.shape_source_xyz-->-0.192<!--/r--> xyz**. Holding the defect fixed, the *reconstructed* surface
  is a **worse** substrate than the real good scan: single-view fusion → Poisson → decimate **smooths away the
  micro-geometry AU-PRO rewards**. The render is faithful; the **mesh is the bottleneck**.
- **The physical defect transfers slightly worse than the grid defect** (twin_phys 0.321 < twin_grid 0.373 on
  the identical xyz substrate) — the Gaussian dent/bump distribution isn't matched to real MVTec defect
  statistics as well as the tuned grid defect. Not loss-engineered toward the eval.
- **Honest negative, competence real.** The Isaac/USD/Replicator pipeline works end-to-end (10 categories
  reconstructed, path-traced, xyz-backprojected, feeding the same rigorous triad) and transfers genuine
  geometric signal — it just doesn't beat on-the-fly synth here.
- **Ablation (Fork A) — reconstruction fidelity is *not* the bottleneck.** Rebuilding every twin at higher
  fidelity (fuse voxel 0.8 → 0.4 mm, decimation cap 150k → 300k faces) → re-render → re-run left shape-source
  **flat** (−0.192 → −0.190 xyz, inside CI): the extra sub-mm mesh detail is resampled away at the 256²
  train/eval resolution. So the reconstructed-surface deficit is a **render-vs-Zivid domain gap** (a clean
  path-traced surface lacks the sensor's structured-light noise), not a mesh-resolution one. The finer mesh
  *did* lift **twin_phys +0.093 (0.321 → 0.414)** — its several-mm physical defects survive 256² where the
  fine grid defect's sub-mm structure does not. Open levers: native-res eval + closing the render↔sensor
  domain gap, not more mesh detail. Full ablation → [`interpretations/twin/2026-07-15_twin-vs-classical.md`](../interpretations/twin/2026-07-15_twin-vs-classical.md).

## Known caveats (the measured gaps to SOTA)
- **Deployable = rgb-only PatchCore, greedy coreset** (<!--r:patchcore.au_pro-->0.902<!--/r-->) — the
  simplest edge path. **3D-geometry fusion is now measured at +1.5pt** (<!--r:fused.au_pro-->0.929<!--/r-->,
  top detector); the remaining gap to SOTA (~0.96) is *better* geometry — native-res FPFH (ours is
  preprocessing-limited, below) or learned point features (M3DM) — not bank or resolution tuning.
- **Our BTF underperforms paper-BTF** (~0.96): we run FPFH on the **256-resized / normalized /
  grazing-noisy** processed cloud; paper-BTF uses **native-resolution clean** point clouds. The
  grazing-angle normal-noise (see `learning/2026-06-25_fpfh-and-neighborhoods.md`) corrupts the
  periphery descriptors.
- AU-PRO here is per-region upper-envelope integrated to FPR 0.3, **self-verified** (perfect = 1.00,
  random = 0.17). Image score = mean of the top-k pixel residuals.

## What this shows (not a leaderboard number)
The result is the **contrast + the measured mechanism**: the intuitive method (reconstruction) *measured*
failing, *diagnosed* (residual ≠ defect signal), and the working paradigm (feature memory bank) reaching
**<!--r:patchcore.au_pro-->0.902<!--/r-->** rgb-only — then **<!--r:fused.au_pro-->0.929<!--/r-->** once 3D-geometry
fusion is added, with the remaining gap to SOTA **named** (better geometry). What this shows is the eval rigor +
the like-for-like comparison, not the single number.

*Reproduce: `python -m surfscan.run vae` (reconstruction) · `... run patchcore` · `... run btf`.
† = verified against the primary source. SOTA row: DCRDF-Net (Wang, Chen, Zhang), *Sensors* 26(2):412,
2026, [doi:10.3390/s26020412](https://doi.org/10.3390/s26020412) — dual-channel reverse-distillation
RGB+3D fusion, mean I-AUROC 0.971 / pixel PRO 0.988. Classic reference: M3DM (CVPR 2023) 0.945/0.964.*


## Drill: where PatchCore's ceiling actually is
Hardened the working method to find the cap, not spray new ones:
- **Bank quality** (greedy k-center coreset + locally-aware features) → **net-flat** (0.908→0.902). The bank wasn't the bottleneck.
- **Resolution** 256→512 → **+1pt** on bagel (0.928→0.937 AU-PRO, img 0.943→0.974). The 256-resize was a *mild* cap — but 4× compute + near-OOM for ~1pt, and it doesn't break the rgb-only ceiling (~0.93).
- **Conclusion:** PatchCore-rgb tops out ~0.93. **Multimodal geometry fusion is the lever — now measured: +1.5pt (<!--r:patchcore.au_pro-->0.902<!--/r-->→<!--r:fused.au_pro-->0.929<!--/r-->) even with our preprocessing-degraded FPFH.** The rest of the gap to SOTA (0.96) is *better* geometry (native-res FPFH / learned point features, M3DM) — NOT bank tuning or rgb resolution. That's the named, measured lever.
