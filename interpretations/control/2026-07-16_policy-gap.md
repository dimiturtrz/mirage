# The control leg's first measured number — a sim-to-real policy gap with a robustness cliff

**Date:** 2026-07-16 · **Epic:** hxq.5 · **Code:** `control/` · **Run:** `python -m control.experiment` (MLflow `control-policy-gap`)

## Question

The control/VLA leg's headline is the same shape as the perception leg's: a **sim-to-real gap**. For
perception it is AU-PRO/AUROC lost when the input distribution shifts; for control it is **task success
lost when the dynamics shift**. This is the leg's first measured instance of that number — end-to-end
through the `core.policy` contract and the `core.rollout` spine, on a fast, reproducible, gate-runnable
task (no Isaac install; that is the next fidelity rung, hxq.3).

## Method — point-mass reach, BC trained in sim only

- **Task.** A planar point mass reaches a goal sampled on the unit circle from rest; success = inside
  `eps=0.15` within a `horizon=40`-step (2 s) episode. Observation is goal-relative `[goal-pos, vel]`.
- **Sim vs real.** A single, physically-argued shift: the real arm carries an **unmodeled payload**
  (mass multiplier) under **actuator sag** (`gain=0.9`, −10% authority). Effective acceleration scales
  by `gain/mass`. We sweep payload +10 … +60%.
- **Policy.** A PD expert (`a = kp·(goal-pos) − kd·vel`, clipped) demonstrates in **nominal sim only**; a
  2-hidden-layer MLP is **behavior-cloned** on those demos. It never sees real.
- **Eval.** 200 matched episodes per condition — sim and every real share goal seeds, so the gap isolates
  the dynamics shift from goal luck — over **5 BC seeds** (net init + demo goals varied), reported mean±sd.
  `gap = success(sim) − success(real)`.

## Result (5 seeds, mean ± sd)

| payload (real) | real success | sim-to-real gap |
|---:|---:|---:|
| +10% | 100.0 ± 0.0% | 0.0 ± 0.0 pp |
| +20% | 100.0 ± 0.0% | 0.0 ± 0.0 pp |
| +30% | 100.0 ± 0.0% | 0.0 ± 0.0 pp |
| +40% | 82.2 ± 5.6% | **17.8 ± 5.6 pp** |
| +50% | 40.0 ± 4.8% | **60.0 ± 4.8 pp** |
| +60% | 2.0 ± 2.8% | **98.0 ± 2.8 pp** |

Expert sim success 100% (achievable ceiling); BC sim success 100.0 ± 0.0% (the clone matches the expert in-domain).

## Interpretation — a robustness margin, then a cliff

The sim-to-real gap here is **not a slope but a threshold**. A closed-loop policy cloned from a PD expert
carries the expert's feedback robustness: up to **+30% payload the gap is exactly zero, with zero variance
across seeds** — feedback absorbs the mismatch and every goal is still reached inside the horizon. Past a
critical payload (~+45%) the actuator can no longer overcome the added inertia within the deadline and
success **collapses**: 82 → 40 → 2% across +40/+50/+60%. The headline is a **60.0 ± 4.8 pp gap at +50%
payload**; the transition is sharp but stable — its spread (±~5 pp) is small next to the jump it sits on.
An earlier signal precedes the cliff: even in the zero-gap zone the mean steps-to-goal climbs (36.5 → 38.6),
so the policy slows *before* it fails.

Why it matters for the engine thesis: the number is produced by the *same* rollout spine and gap metric a
higher-fidelity env will use — the `Env` Protocol is the seam. Swapping the numpy point mass for an Isaac
Lab env (hxq.3) re-measures this exact curve at higher fidelity **without touching the policy, the spine,
or the metric**. The mechanism (robustness margin → cliff) is the transferable finding; the numbers will
sharpen with fidelity.

## Integrity

- **Not Isaac.** A numpy point mass is a deliberate first rung — fast, deterministic, CI-runnable — a
  *projection* of the sim-to-real mechanism, not a high-fidelity robot. Isaac (installed, in `sim/`'s
  isolated env) is the next rung; this result is honest about being the toy-fidelity anchor.
- **5 seeds, reproducible.** Each BC seed varies net init + demo goals; the run is deterministic (torch +
  env + eval all seeded). The ±sd is the seed spread — small (≤5.6 pp) relative to the transition, so the
  margin-then-cliff shape is not single-init scatter. More seeds would tighten the exact transition payload.
- **The shift is chosen, not fit.** Payload + actuator-sag are mechanical, argued a priori; we sweep the
  whole range rather than cherry-pick one shift, so the reported gap is a point on a disclosed curve.

## Reproduce

```bash
python -m control.experiment      # prints the table; logs params+metrics to MLflow 'control-policy-gap'
```
