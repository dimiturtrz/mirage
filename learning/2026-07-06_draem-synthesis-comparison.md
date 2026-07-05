# DRAEM synthesis comparison — my synth lost to Perlin, and I over-claimed why

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

## Meta-point
The eval spine did its job twice: it refuted my "realism → better" assumption **and** exposed a
cherry-picked number in the repo. Then I over-generalized the refutation into a mechanism claim — and
that got caught too. The discipline isn't "measure once and conclude"; it's "measure, then don't claim
more than the measurement supports."
