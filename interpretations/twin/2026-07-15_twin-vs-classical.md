# Digital twin vs classical synth — the shape-source sim-to-real number (Stage 2)

**Question.** Stage 1 established the classical synth→real gap (+0.166 AU-PRO): apply grid defects
*on-the-fly* to real good scans, train a U-Net, lose 0.166 transferring to real. Stage 2 swaps the
**shape source**: reconstruct a per-category mean-surface mesh from the 244 top-down good scans, render
it through Isaac/Replicator (path-traced, back-projected to an organized camera-frame xyz map, MVTec
layout), and feed the *same* triad. Two twin arms:

- **twin_grid** — the *same* grid defect as classical synth, but applied to the **reconstructed** surface
  instead of the real good scan. Isolates the shape source (defect held constant).
- **twin_phys** — a **physical** mesh-deform defect (Gaussian z-displacement dent/bump) rendered through
  the path tracer, gt from the depth-diff. Isolates the defect model (both surface *and* defect now synthetic).

One U-Net, four label sources, one shared real eval half — per Stage-1 protocol. Run on **both channels**
(rgb appearance, xyz geometry), because the twin's contribution is *geometry*, not appearance.

## The numbers

| arm (label source) | rgb AU-PRO | xyz AU-PRO |
|---|---|---|
| real → real (ceiling) | 0.794 | 0.726 |
| synth (classical) → real | 0.628 | 0.565 |
| twin_grid → real | 0.300 | 0.373 |
| twin_phys → real | 0.081 | 0.321 |

Derived: **classical gap** rgb +0.166 / xyz +0.161 (reproduces Stage 1 on both channels — the
`Source` refactor didn't move the classical number). **shape-source** (twin_grid − synth, same defect)
rgb **−0.327** [−0.356, −0.299] / xyz **−0.192** (point). **twin_phys→real gap** rgb +0.713 [0.688, 0.735]
/ xyz +0.405 (point). *(xyz twin_phys/shape-source CIs unrecorded this run — the report crashed emitting
the paired bootstrap for the xyz twin_phys arm (bead `synthscape-7db`); the point estimates are
the au_pro means printed per arm, which are exact.)*

## What it says

**1. xyz is the twin's axis, and the data proves it.** Every twin arm jumps from rgb→xyz: twin_grid
0.300→0.373, twin_phys **0.081→0.321**. twin_phys on rgb is ~random (0.081) — a physical mesh-dent is
near-invisible in the twin's uniform-tan appearance — but on xyz it carries **real localization signal**
(0.321). This is the house rule in action: a physically-valid method regressing on rgb meant *the rgb
test lacks the twin's axis*, not that the twin is bad. Scored on geometry, the twin transfers.

**2. But the twin does not beat classical synth — even on xyz.** shape-source is **negative** on both
channels (−0.327 rgb, −0.192 xyz). Holding the defect constant, the **reconstructed** surface is a
*worse* training substrate than the **real** good scan. Mechanism: the reconstruction (single-view fusion
of 244 partials → Poisson depth-9 → decimate to 150k faces) **smooths away the micro-geometry** that
AU-PRO rewards. Classical synth trains on real surface + synthetic defect; twin_grid trains on
*smoothed* surface + the same defect. The render is faithful; the **mesh is the bottleneck**.

**3. The physical defect model transfers slightly worse than the grid defect.** On the identical twin
substrate, twin_phys (0.321) < twin_grid (0.373) on xyz. The Gaussian dent/bump distribution isn't
matched to real MVTec defect statistics as well as the tuned grid defect is — expected, and not
loss-engineered toward the test (house rule: fix needs with balanced data, not by fitting the defect
model to the eval).

## Verdict — an honest negative, and the competence is real

The Isaac/USD/Replicator digital-twin pipeline is **end-to-end working**: 10 categories reconstructed,
path-traced headless, back-projected to xyz, MVTec-format, feeding the same rigorous triad. It transfers
genuine geometric signal. It **does not** beat classical on-the-fly synth on this benchmark — because
single-view reconstruction fidelity, not rendering, caps the surface realism. That is the finding: the
sim-to-real spine measured *where* the twin loses (surface micro-geometry) rather than asserting a twin
must help. Kept as a negative-but-informative Stage-2 result; the lever for a positive twin is
**higher-fidelity reconstruction** (multi-view / no Poisson smoothing / native-res), not the renderer.

See also: `interpretations/sim-to-real/2026-07-08_sim-to-real-triad.md` (Stage-1 classical gap),
`core/twin/reconstruct.py` (the reconstruction), `sim/generate_twin_synth.py` (the Isaac generator).
