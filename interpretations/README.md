# interpretations/

Sense-making of **our own** results — reading across runs and artifacts (MLflow, `predictions.npz`,
the eval harness) to say *what a number means and why*, not just what it is. One of the three doc layers
(see [`../CLAUDE.md`](../CLAUDE.md)):

- [`../research/`](../research/) — external / field synthesis (theirs: SOTA, benchmarks, method surveys).
- [`../learning/`](../learning/) — the study ramp, general understanding (theory + quiz logs).
- **`interpretations/`** — *our* measured results, interpreted. Split **per task**; `converging/` holds
  cross-task reads once a claim spans more than one.

Layout: `interpretations/<task>/<date>_<topic>.md`. The headline lives in the portfolio README /
[`../docs/RESULTS.md`](../docs/RESULTS.md); the interpretation here carries the mechanism + the diagnosis
it links out to.

## Tasks
- **`detector/`** — Stage-0 anomaly detector (reconstruction vs feature-space, memory banks).
  - `2026-07-06_feature-space-vs-pixel-reconstruction.md` — why pixel-residual is ≈ random and
    feature-space localizes (the 0.095 → 0.902 finding).
- **`synthesis/`** — synthetic-defect generation (DRAEM, realism vs signal).
  - `2026-07-06_draem-synthesis-comparison.md` — my synth lost to Perlin on a broken displacement axis,
    fixed (normal-displaced), now tied on localization / better on detection.
- **`sim-to-real/`** — Stage-1 transfer (the triad, the measured gap, domain adaptation).
  - `2026-07-08_sim-to-real-triad.md` — method + diagnosis behind the 0.166 gap and the AdaBN negative.
