# DRAEM synthesis comparison — my synth lost to Perlin (broken axis), fixed, now tied/better

**Status: RESOLVED 2026-07-07** (see bottom). v2 lost because it displaced along world-z; v3 displaces
along the surface normal → gap closed. The arc below is kept intact as the honesty record.

**2026-07-06.** Built channel-aware "realistic" defect synthesis for Stage-1 classical, expecting it to
beat DRAEM's crude Perlin. It lost. I first wrote this up as *"realism hurts DRAEM"* — that was
**overreach**, caught on review. This note states the honest, narrow version.

## What was actually measured (bagel/xyz, 150 ep, 3 seeds each)
| synthesis | au_pro (seeds 0/1/2) | mean |
|---|---|---|
| crude Perlin | 0.481 · 0.302 · 0.295 | **0.36 ± 0.09** |
| my "realistic" (channel-aware, z-displacement + rgb appearance) | 0.157 · 0.197 · 0.186 | **0.18 ± 0.02** |

Bands don't overlap → **my v2 synth reliably underperformed Perlin for DRAEM-on-xyz.** That is the
only established fact.

## What is NOT established — the cause
I claimed the mechanism ("realism starves a discriminator of a strong target"). That's a **hypothesis
dressed as a result.** My "realistic" synthesis is dubious enough that the loss is at least as likely a
**flawed synth** as a law about realism:
- **z-displacement ≠ surface-normal displacement.** A real dent pushes along the surface normal; I
  pushed along the z-axis. On the bagel's curved sides the normal is ~radial → a z-push there is a
  **lateral shear**, not a dent. Geometrically wrong over most of the object.
- **Magnitude 15–35% of the object z-range** = a giant gouge, not a subtle defect.
- **Tested xyz-only** → the rgb appearance edits (the more defensible half) never ran.

So "realistic" was a generous label for "smooth z-blob, wrong axis, arbitrary size." Losing to Perlin
may just mean *my geometry is broken*, not that realism hurts DRAEM.

## The second finding (this one holds — honesty fix)
Perlin **seed 0 = 0.481** reproduces the historical RESULTS.md "0.48" exactly; seeds 1,2 ≈ 0.30. **The
published 0.48 was a single lucky seed;** honest multi-seed mean ~0.36. Corrected in the repo. Multi-seed
before any synthesis claim — single-seed DRAEM swings ±0.09.

## To disentangle (owed)
1. **Visualize** a synthetic bagel — is it a plausible dent or a shear-crater? If garbage → it's my synth.
2. **Normal-displacement, sensible magnitude** (compute normals from the geometry) — a real dent.
3. **Vary contrast** — if a realistic-*shaped* high-contrast defect also loses, then shape not contrast.

## RESOLUTION (2026-07-07) — it was my broken geometry, not realism
Did owed item #2: replaced z-displacement with **true surface-normal displacement** (new
`core/geometry.py` grid_normals — per-pixel normals off the position map; one-sided-diff fallback for
sensor dropouts, 95% coverage). Re-ran the *same* clean 3-seed bagel/xyz sweep:

| synthesis | au_pro (seeds 0/1/2) | au_pro mean | img_auroc mean |
|---|---|---|---|
| crude Perlin | 0.481 · 0.302 · 0.295 | 0.36 ± 0.09 | 0.61 |
| realistic **v3 (normal-displaced)** | 0.400 · 0.323 · 0.232 | **0.32 ± 0.07** | **0.79** |

- **au_pro: now a TIE** (0.32 vs 0.36 — gap 0.04 < both stds, bands overlap). The v2 deficit
  (0.18 ≪ 0.36, non-overlapping) is **gone.** Fixing the displacement *axis* closed the whole gap.
- **img_auroc: realistic v3 WINS** — 0.79 vs 0.61, and realistic > Perlin on **all 3 seeds**
  (consistent → real signal, not scatter). Normal-displaced defects give a better whole-image signal.
- Perlin seed-0 = 0.481 **again** the lucky outlier; drop it and Perlin au_pro ≈ 0.30 = level with v3.

**Verdict:** the v2 loss was **flawed synth (wrong displacement axis), not a law that realism hurts
DRAEM** — exactly the alternative hypothesis flagged above, now confirmed. Realistic-normal went from
*clearly worse* to *tied on localization, better on image-detection*. Owed #1 (visualize) and #3 (vary
contrast) remain open but are no longer needed to explain the v2 result.

## Meta-point
The eval spine did its job **three** times: refuted "realism → better", exposed a cherry-picked repo
number, and now refuted my *over*-correction ("realism hurts") by fixing the real bug. The discipline
isn't "measure once and conclude"; it's "measure, hold the claim to the measurement, then fix the actual
defect and re-measure." Physics (displace along the normal) beat the statistical story both times.
