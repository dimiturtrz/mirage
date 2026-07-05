# Realism hurts DRAEM — a discriminator wants a strong signal, not a real one

**2026-07-06.** Stage-1 classical: built realistic channel-aware defect synthesis to replace DRAEM's
crude Perlin blobs, expecting better localization. **Measured the opposite** — and multi-seed control
caught both a wrong assumption and a cherry-picked number in our own results.

## The measurement (bagel/xyz, 150 ep, 3 seeds each)
| synthesis | au_pro (seeds 0/1/2) | mean |
|---|---|---|
| crude Perlin | 0.481 · 0.302 · 0.295 | **0.36 ± 0.09** |
| realistic (channel-aware, smooth z-displacement + rgb appearance) | 0.157 · 0.197 · 0.186 | **0.18 ± 0.02** |

Bands don't overlap (Perlin min 0.295 > realistic max 0.197). **Realistic synthesis is reliably worse
for DRAEM.**

## Why (the mechanism)
DRAEM trains a **discriminator** to segment the synthetic defect, then applies it to real defects.
A discriminator needs a **strong, learnable target**:
- Crude Perlin = high-frequency, high-contrast → an easy, sharp segmentation target → strong gradient.
- Realistic physical defect = smooth, subtle depression → a weak target → the disc learns less.

So **realism (subtlety) starves the discriminator.** The DRAEM paper itself blends aggressive DTD
textures, not subtle physical defects — consistent with this. "More realistic training data" is *not*
universally better; it depends on what the method needs. For a discriminative synthesis-detector,
contrast beats fidelity.

## The second finding (honesty fix to our own number)
Perlin **seed 0 = 0.481** reproduces the historical RESULTS.md "0.48" exactly — but seeds 1,2 = ~0.30.
**The published 0.48 was a single lucky seed;** the honest multi-seed mean is 0.36. We committed the
exact single-seed sin the project preaches against — caught only by running the control. RESULTS.md
corrected.

## Consequences
1. **Keep crude/high-contrast synth for DRAEM** — don't feed it subtle-realistic defects.
2. **Realistic synthesis's payoff is elsewhere**: the sim-to-real **transfer** experiment (train a
   *non-discriminative* detector on synth, test on real, where fidelity→transfer genuinely matters) and
   the **Stage-2 digital twin**. DRAEM was the wrong first testbed.
3. **Multi-seed before any synthesis-quality claim.** Single-seed DRAEM swings ±0.09 — enough to invent
   or erase an effect.

## The meta-point
This is the honest-eval spine working on *itself*: it refuted my "realism → better" assumption **and**
exposed a cherry-picked number already in the repo. The value wasn't the synthesis — it was measuring
that it didn't do what I assumed.
