# The sim-to-real triad — the first honest gap number (Stage 1)

Field-map topic #8. Stage 0 asked *can we detect defects?* (yes — PatchCore 0.91). Stage 1 asks the
harder, honest question: **if you train a detector on SYNTHETIC defects, how much does it lose on REAL
ones — and can you close it?** "I modeled it" never stands in for "I measured it transferred." This note
is that measurement.

## The design — isolate one variable
Three arms, **one** architecture (a supervised defect-segmentation U-Net — DRAEM's disc net, no recon
branch to confound), scored on **one** shared real eval set. Only the **training-label source** changes:

| arm | trains on | is |
|---|---|---|
| **real → real** | real defect masks (calib half of the test split) | the **ceiling** |
| **synth → real** | our synthetic defects painted on good scans | the **gap** |
| **synth+DA → real** | synth-trained, then AdaBN on real (unlabeled) | the **closure** |

Because the arch, eval set, and epochs are identical, the *only* thing that moves the number is where the
labels came from — so the drop IS the sim-to-real gap, not an architecture artifact.

**No leakage:** the test split is partitioned per-label into `calib` (real arm trains here) and `eval`
(everyone scores here) — disjoint, stratified so both carry good + defective (unit-tested). The synth arm
never touches real defects at all (trains on the good *train* split + synthesized labels).

## The gap (`real - synth`)
Training on synthetic defects loses a large, real chunk of localization vs real labels. It's the honest
headline — the number the easy demo (train on real, report a flattering score) hides.

## AdaBN — the domain-adaptation arm (closure)
The synth→real drop is largely **distribution shift**: a synth-trained net sees real-scan statistics
(sensor noise, real texture — see the structured-light lesson) it never trained on, and its BatchNorm
running stats are wrong for that domain. **AdaBN** (Li 2017) fixes exactly that: reset BN running stats,
recompute them from the real (unlabeled) images via forward passes — no labels, no backprop. Cheap,
standard, unsupervised. Transductive (adapts on the eval images it scores — statistics only).

## Calibration under shift (ECE) — the contribution layer
A synth-trained model isn't just less accurate on real — it's **over-confident**: its probabilities don't
match reality. Pixel-level **ECE** (Expected Calibration Error) measures that mismatch. The finding: ECE
rises sharply under the synth→real shift (miscalibrated), and **AdaBN restores it to ceiling level** — the
same BN realignment that recovers localization also recovers trust. This is why "calibration under shift"
is a headline metric, not an afterthought.

## The closed-loop curriculum — the differentiator (carried)
The acoustic engine adapts generation difficulty per epoch from the trainer's score. Ported here as
`KindCurriculum`: track an EMA of training loss per defect **kind**, sample the next batch's kind
∝ softmax(standardized loss) — the generator chases whatever kind the detector is currently worst on,
instead of a fixed uniform mix. Single-kind batches so the scalar loss attributes cleanly to one kind.

## RESULTS (bagel..all-10, rgb, 100 epochs, 3 seeds)
<!-- fill from `python -m surfscan.experiments.triad_summary` once the curriculum seeds land -->

| arm | au_pro (3-seed) | img_auroc | ECE |
|---|---|---|---|
| real → real (ceiling) | 0.804 ± 0.007 | 0.892 | 0.011 |
| synth → real (gap) | 0.537 ± 0.049 | 0.635 | 0.037 |
| synth+DA → real (AdaBN) | 0.660 ± 0.045 | 0.579 | 0.013 |
| synth + curriculum | _pending seeds 1,2_ | | |

- **Sim-to-real GAP ≈ 0.27 au_pro (27 pp).** Ceiling rock-stable (±0.007), synth wider (±0.049).
- **AdaBN closes ~46%** (+0.12) **and restores calibration** (ECE 0.037 → 0.013 ≈ ceiling). But it
  *hurts image-detection* (img 0.64 → 0.58) — helps where the pixels are, costs whole-image scoring. An
  honest trade, reported not hidden.
- **Real ceiling 0.80 < PatchCore 0.91 (unsupervised).** Supervised-from-scarce-labels loses to the
  memory bank — the real arm is a *scarce-label* ceiling (~half a small defect set), not an oracle.
- **Curriculum:** _to fill_ (seed-0 was +0.019, likely marginal — report honestly either way).

## The honest takeaways
- The gap is **big and measured**, not modeled-and-assumed. That's the Stage-1 deliverable.
- A cheap, label-free DA method (AdaBN) recovers ~half of it **and** fixes calibration — the calibration
  win is arguably the more interesting result than the localization one.
- Synthetic data's real payoff here isn't beating real labels — it's that real labels are *scarce*
  (supervised real→real < unsupervised PatchCore), so unlimited synthetic labels + DA is a credible path.
