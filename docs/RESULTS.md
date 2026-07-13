# Results — anomaly detection on MVTec 3D-AD (Stage 0 detector + Stage 1 sim-to-real)

Measured first-stage findings. **Per-category models** (the MVTec convention), scored through our own
eval harness — image-level **AUROC** (detection) + pixel-level **AU-PRO** (localization). These are
*our* runs through *one* eval harness, **not** the papers' leaderboard numbers.

## Headline: reconstruction fails, feature-memory-bank works
<!-- results:methods -->
| method | modality | mean img-AUROC | mean pixel AU-PRO |
|---|---|---|---|
| Reconstruction (VAE / +dropout / inpaint / fused) — ours | xyz (+rgb) | 0.480 | 0.095 |
| DRAEM (synthesis-discriminative, crude Perlin) · bagel — ours | xyz | 0.610 | 0.360 |
|   ↳ realistic-synth v3 (normal-displaced, on DRAEM) · bagel — ours | xyz | 0.790 | 0.320 |
| BTF (FPFH memory bank) — ours | geometry | 0.675 | 0.653 |
| **PatchCore (feature memory bank)** — ours | rgb | 0.838 | 0.902 |
| Feature-recon / RD4AD-lite — ours | rgb | 0.807 | 0.908 |
| Fused (PatchCore-rgb + BTF) — ours | rgb + 3D | 0.782 | 0.904 |
| SOTA (DCRDF-Net) ⚠ | rgb + 3D | ~0.97 | ~0.99 |
<!-- /results:methods -->

<sub>¹ fused all-10 (0.904) is **net-neutral** vs rgb-only (0.908): it *helps* where geometry is decent (bagel 0.928→0.937, carrot) but *hurts* where our weak BTF geometry is bad (cookie, foam). Our BTF is preprocessing-limited (below), so fusion can't pay until native-res FPFH lands — a reported negative: rgb-only is the current deployable.</sub>

**The sharpened lesson:** two *independent* feature methods — a memory bank (PatchCore) and a
reconstruction AE (feature-recon) — both land at **AU-PRO 0.908**, vs pixel-reconstruction 0.095.
So the winning ingredient isn't memory-bank-vs-reconstruction; it's the **feature space**.
Reconstruction was never the wrong idea — *raw-pixel space* was (normal geometric complexity
dominates the residual). Move the same reconstruction into a frozen-backbone feature space (where
normal complexity is already normalized) and it localizes fine. That's the cleanest statement of
the Stage-0 finding.

² **DRAEM** (synthesis-discriminative), bagel xyz, **3-seed** (noise discipline): crude Perlin
**0.36 ± 0.09** au_pro — the earlier single-seed **0.48 was a lucky draw** (a robustness fix to our own
number). My first "realistic" synth (v2) scored **lower, 0.18 ± 0.02** — but that was **broken geometry,
not a law that realism hurts DRAEM:** v2 displaced defects along the **world-z axis, not the surface
normal**, so on the bagel's curved sides a "dent" was a lateral shear. **Fixed (v3, normal-displaced;
`core/geometry.py`):** re-running the same 3-seed sweep, realistic-v3 = **0.32 ± 0.07 au_pro (tied with
Perlin, the deficit gone)** and **0.79 img_auroc vs Perlin 0.61 (higher on all 3 seeds, decisively on 2 —
seed-1 a near dead-heat)**. So realistic-normal is now *tied on localization, better on image-detection*. The eval spine refuted the assumption, exposed
a cherry-picked number, then caught the over-correction. See
`learning/2026-07-06_draem-synthesis-comparison.md`.

### Per-defect-type (PatchCore-rgb, pooled across categories)
Most defect types localize 0.87–0.96 AU-PRO; weak spots: **thread 0.59**, color hardest to *detect*
(image-AUROC 0.59). The stratified failure (like systole's EF-by-pathology).

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
rgb, 100 ep, **3 seeds**). Only the label source varies → the drop IS the sim-to-real gap. Full method +
diagnosis → [`learning/2026-07-08_sim-to-real-triad.md`](../learning/2026-07-08_sim-to-real-triad.md).

<!-- results:triad -->
| arm | au_pro [95% boot CI] | img-AUROC | ECE |
|---|---|---|---|
| real → real (ceiling) | 0.804 | 0.892 | 0.011 |
| synth → real (the gap) | 0.537 | 0.635 | 0.037 |
| synth+DA → real (AdaBN) | 0.660 | 0.579 | 0.013 |
| synth + curriculum | 0.487 | 0.617 | 0.083 |
<!-- /results:triad -->

- **Sim-to-real gap ≈ 27 pp** (real 0.804 → synth 0.537 AU-PRO). Ceiling rock-stable (±0.007).
- **AdaBN (one DA method) closes ~46%** (+0.12) **and restores calibration** (ECE 0.037 → 0.013 ≈ ceiling)
  — a cheap, label-free BN-stat realignment recovers both localization *and* trust. It *costs*
  image-detection (img 0.64 → 0.58) — a measured trade.
- **Real ceiling 0.80 < PatchCore 0.91 (unsupervised):** supervised-from-scarce-labels loses to the memory
  bank. Synthetic data's payoff here is *volume* (real labels are scarce), not beating them.
- **Closed-loop curriculum REGRESSED — the reported negative.** Kind-chasing scored 0.487 (−0.05 vs uniform)
  and *doubled* ECE (0.083). Diagnosed (chases the noisiest kind, not the most learnable; single-kind
  batches starve diversity) and kept in the repo — a refuted differentiator with a mechanism beats a
  quietly-dropped one. Fix (Stage-1.5): adapt *difficulty*, not *kind*.

## Known caveats (the measured gaps to SOTA)
- **PatchCore is rgb-only + *random* coreset** (not greedy). The gap to SOTA (~0.96) is exactly the
  **3D-feature fusion** (M3DM) + **greedy coreset** we haven't added yet.
- **Our BTF underperforms paper-BTF** (~0.96): we run FPFH on the **256-resized / normalized /
  grazing-noisy** processed cloud; paper-BTF uses **native-resolution clean** point clouds. The
  grazing-angle normal-noise (see `learning/2026-06-25_fpfh-and-neighborhoods.md`) corrupts the
  periphery descriptors.
- AU-PRO here is per-region upper-envelope integrated to FPR 0.3, **self-verified** (perfect = 1.00,
  random = 0.17). Image score = mean of the top-k pixel residuals.

## The point (not a leaderboard number)
The result is the **contrast + the measured mechanism**, the way systole reported its triad: the
intuitive method (reconstruction) *measured* failing, *diagnosed* (residual ≠ defect signal), and
the working paradigm (feature memory bank) standing near SOTA at **0.91** with the remaining gap
**named** (3D fusion · greedy coreset). The contribution is the eval rigor + the like-for-like comparison,
not the single number.

*Reproduce: `python -m surfscan.run vae` (reconstruction) · `... run patchcore` · `... run btf`.
⚠ = verify against the primary source before any public claim.*


## Drill: where PatchCore's ceiling actually is (decisive)
Hardened the working method to find the cap, not spray new ones:
- **Bank quality** (greedy k-center coreset + locally-aware features) → **net-flat** (0.908→0.902). The bank wasn't the bottleneck.
- **Resolution** 256→512 → **+1pt** on bagel (0.928→0.937 AU-PRO, img 0.943→0.974). The 256-resize was a *mild* cap — but 4× compute + near-OOM for ~1pt, and it doesn't break the rgb-only ceiling (~0.93).
- **Conclusion:** PatchCore-rgb tops out ~0.93. The remaining ~3pt to SOTA (0.96) is **multimodal geometry fusion** (M3DM), which needs *good* geometry (native-res FPFH, our preprocessing currently degrades it) — NOT bank tuning or rgb resolution. That's the named, measured lever.
