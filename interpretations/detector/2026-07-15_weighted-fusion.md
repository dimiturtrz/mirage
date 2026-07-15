# Weighted fusion beats equal-sum — the geo bank helps *localize*, not *detect*

The prior verdict was that fusing geometry (FPFH) into rgb PatchCore was "net-neutral" — not worth it.
That was wrong, and wrong for a specific, learnable reason: the fusion was an **equal-weighted sum** of
the two z-scored map families, and equal weight sits in the *valley* between the two modalities' optima.
Re-weighting toward the stronger bank turns the same two banks into a Pareto win.

## The setup
Two frozen memory banks, no training: rgb PatchCore (strong) + FPFH geometry (weak alone). Each emits a
per-pixel anomaly map; z-score each over the valid test pixels, combine, score. The only free knob is the
combination weight `w`:  `fused = w·z(rgb) + (1−w)·z(geo)`.  `w=0.5` = the old equal-sum, `w=1` = rgb-alone.

## What the sweep showed (10 cats, w ∈ [0,1])
A fused optimum at **w < 1 beats rgb-alone on au_pro for 10/10 categories**. The per-category optimal
`w` spans **0.3–0.9** — which is exactly why a single equal weight underperforms: it can't sit on each
category's peak, and `w=0.5` lands below rgb-alone on the *detection* metric.

Held-out check (w=0.7 locked on 4 dev cats, tested on the untouched 6): au_pro macro **0.911 → 0.943
(+3.1pp), 6/6 cats** — the headroom is systematic, not a fit to one category.

## The mechanism — localization vs detection
The gain is **not uniform across the two headline metrics**, and the split is the interesting part:

- **au_pro (localization) improves** — geometry sharpens *where* the defect is. A dent or warp is a
  geometric event the rgb bank sees only as faint shading; the FPFH bank lights it up crisply.
- **img_auroc (detection) does not** — the rgb bank already answers *whether* a defect exists; adding a
  noisier geo score to the image-level statistic can only dilute it. Equal-sum (`w=0.5`) actively *hurt*
  img_auroc (0.822 < PatchCore 0.838); pulling `w` back to 0.7 recovers it.

So the honest framing: **fusion buys localization, and the weight is what stops it from costing
detection.** Geometry is a *where* signal grafted onto an rgb *whether* signal.

## The number (10-cat macro, 95% bootstrap CI)
| method | au_pro | img_auroc |
|--------|--------|-----------|
| PatchCore (rgb-alone) | 0.902 [0.892, 0.910] | 0.838 [0.812, 0.866] |
| Fused, equal-sum (old) | 0.917 [0.911, 0.923] | 0.822 |
| **Fused, weighted w=0.7 (new)** | **0.929 [0.923, 0.935]** | **0.840 [0.813, 0.868]** |

Weighted fusion **Pareto-dominates the old equal-sum fusion** (+1.2pp au_pro, +1.8pp img_auroc) from a
one-line weight change, and lifts au_pro +2.7pp over rgb-alone with detection held flat. The new au_pro
lower bound (0.923) meets the old upper bound — the marginal CIs are essentially disjoint.

## Honest caveats
- `w=0.7` is a **single global constant**, justified as down-weighting the empirically weaker geo bank
  ~2:1 — not a per-category fit. The per-cat optimal `w` (0.3–0.9) means real headroom is left on the
  table: the peeked best-`w` macro au_pro is ~0.95.
- This is **not** the 0.96 SOTA target. It is the honest, CI-backed increment a *weighted* linear fusion
  buys. Closing the rest needs a **learned per-category weight** — M3DM's OCSVM decision-fusion, which
  picks the combination unsupervised from train-good and should capture the per-cat `w`-variance without
  the global-constant detection tradeoff. That is the next step (bead iwq).
