# Stage 0 results — anomaly detection on MVTec 3D-AD

Honest first-stage findings. **Per-category models** (the MVTec convention), scored through our own
eval harness — image-level **AUROC** (detection) + pixel-level **AU-PRO** (localization). These are
*our* runs through *one* eval harness, **not** the papers' leaderboard numbers.

## Headline: reconstruction fails, feature-memory-bank works
<!-- results:methods -->
| method | modality | mean img-AUROC | mean pixel AU-PRO |
|---|---|---|---|
| Reconstruction (VAE / +dropout / inpaint / fused) — ours | xyz (+rgb) | 0.480 | 0.095 |
| DRAEM (synthesis-recon, crude Perlin) — ours | xyz / rgb | — | 0.480 |
| BTF (FPFH memory bank) — ours | geometry | 0.675 | 0.653 |
| **PatchCore (feature memory bank)** — ours | rgb | 0.819 | 0.908 |
| Feature-recon / RD4AD-lite — ours | rgb | 0.807 | 0.908 |
| Fused (PatchCore-rgb + BTF) — ours | rgb + 3D | 0.782 | 0.904 |
| SOTA (M3DM) ⚠ | rgb + 3D | ~0.94 | ~0.96 |
<!-- /results:methods -->

<sub>¹ fused all-10 (0.904) is **net-neutral** vs rgb-only (0.908): it *helps* where geometry is decent (bagel 0.928→0.937, carrot) but *hurts* where our weak BTF geometry is bad (cookie, foam). Our BTF is preprocessing-limited (below), so fusion can't pay until native-res FPFH lands — an honest negative: rgb-only is the current deployable.</sub>

**The sharpened lesson:** two *independent* feature methods — a memory bank (PatchCore) and a
reconstruction AE (feature-recon) — both land at **AU-PRO 0.908**, vs pixel-reconstruction 0.095.
So the winning ingredient isn't memory-bank-vs-reconstruction; it's the **feature space**.
Reconstruction was never the wrong idea — *raw-pixel space* was (normal geometric complexity
dominates the residual). Move the same reconstruction into a frozen-backbone feature space (where
normal complexity is already normalized) and it localizes fine. That's the cleanest statement of
the Stage-0 finding.

² **DRAEM** (synthesis-recon) is weak with *crude* Perlin-noise synthesis (bagel xyz 0.48, rgb 0.21):
the discriminator learns to spot noise, not defects. This is the bottleneck DAS3D/3DSR fix with
realistic synthesis — so it **experimentally motivates the Stage-1 engine**: realistic synthetic
defects → DRAEM discriminator → the SOTA path. The project's two halves (synthesis + detection)
connect here.

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
<!-- /results:per_category -->

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

*Reproduce: `python -m surfscan.experiments.run_all` (reconstruction) · `... run_patchcore` · `... run_btf`.
⚠ = verify against the primary source before any public claim.*


## Drill: where PatchCore's ceiling actually is (decisive)
Hardened the working method to find the cap, not spray new ones:
- **Bank quality** (greedy k-center coreset + locally-aware features) → **net-flat** (0.908→0.902). The bank wasn't the bottleneck.
- **Resolution** 256→512 → **+1pt** on bagel (0.928→0.937 AU-PRO, img 0.943→0.974). The 256-resize was a *mild* cap — but 4× compute + near-OOM for ~1pt, and it doesn't break the rgb-only ceiling (~0.93).
- **Conclusion:** PatchCore-rgb tops out ~0.93. The remaining ~3pt to SOTA (0.96) is **multimodal geometry fusion** (M3DM), which needs *good* geometry (native-res FPFH, our preprocessing currently degrades it) — NOT bank tuning or rgb resolution. That's the named, measured lever.
