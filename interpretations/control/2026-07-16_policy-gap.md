# The control leg — a sim-to-real policy gap: the cliff, two levers (DR vs adaptive), and a PhysX rung

**Date:** 2026-07-16 · **Epic:** hxq · **Code:** `control/` + `sim/isaac_reach.py` · **Run:** `python -m control.experiment` (point mass — nominal/DR/adaptive) · `sim/ … isaac_gap.py` (PhysX rung)

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
- **Lever 1 — domain randomization.** A second arm clones the same (blind) expert on demos whose dynamics
  are **randomized per episode** — payload mass `~U(1.0, 1.6)` and actuator gain `~U(0.85, 1.0)`, covering
  the test-shift range. Same expert, same spine, same eval (nominal sim + shifted reals); only demo-collection
  changes. Goals still ride the seed, so nominal-vs-DR is matched per seed.
- **Lever 2 — adaptive policy (online system-ID).** A third arm gives the policy a *proprioceptive cue*: the
  observation is augmented with `[last_action, Δv]` (a `ProprioEnv` wrapper), and the demonstrator is an
  online controller that estimates the plant gain `k = gain/mass` from `‖Δv‖ / (dt·‖last_action‖)` and
  inverts it — `a = clip((kp·e − kd·v) / k̂)` — so a heavier payload is met with a proportionally larger
  command. Cloned under DR, from **observable** proprioception only (no privileged `Phys`). This is the fix
  the DR analysis pointed to; the arm tests whether *sensing* the payload beats merely *randomizing* over it.
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

## Two levers — randomize over the shift, or sense it (5 seeds, mean ± sd)

Two ways to move the cliff, tested side by side against the nominal (blind, nominal-demo) baseline:

| payload (real) | nominal gap | **DR** gap | **adaptive** gap |
|---:|---:|---:|---:|
| +10 … +30% | 0.0 ± 0.0 pp | 0.0 ± 0.0 pp | 0.0 ± 0.0 pp |
| +40% | 17.8 ± 5.6 pp | 7.7 ± 4.4 pp | **0.0 ± 0.0 pp** |
| +50% | **60.0 ± 4.8 pp** | 56.0 ± 1.2 pp | **0.0 ± 0.0 pp** |
| +60% | 98.0 ± 2.8 pp | 93.3 ± 3.3 pp | **18.1 ± 6.9 pp** |

**Domain randomization is a partial lever.** It halves the gap at the cliff *onset* (+40%: 17.8 → 7.7 pp) and
tightens the seed spread, but barely dents the deep collapse (+50/+60% stay > 55 pp). The reason is
mechanistic: the blind policy's observation is `[goal-pos, vel]` with **no cue of payload**, so DR cannot
teach it to *sense* and compensate the shift — it can only widen the state distribution the clone is fit on,
buying robustness in the marginal band. Past the actuator's authority limit, more demo variety can't help.

**The adaptive policy nearly closes the gap.** Giving the policy one step of proprioception — the last action
and the velocity change it produced — lets it identify the plant gain online and invert it, and that
**erases the cliff up to +50% payload (60.0 → 0.0 pp)** and cuts the extreme +60% case from 98 → 18 pp. This
is the decisive contrast the leg was built to draw: **the fix for a dynamics shift is to *observe* it, not to
average over it in training.** DR spreads the training distribution; the adaptive policy closes the loop. What
remains at +60% is the honest residue — even a perfectly-estimated inverse can't manufacture force beyond the
`amax` clip, so when `gain/mass` is small enough the compensated command saturates and a fraction of goals
still miss the deadline. A real physical ceiling, not an estimator failure.

**Caveat — why this closes so cleanly.** The toy plant is a clean scalar-gain LTI system, so one-step online
system-ID is easy and near-exact; on a real robot (nonlinear, higher-dimensional, sensor-noisy) that
identification is far harder and the recovery would be partial. The transferable claim is directional —
*proprioceptive/adaptive observation is the right lever, and it dominates randomization* — not "system-ID
solves sim-to-real". Testing how much of this survives at higher fidelity is exactly what the seam is for.

Why it matters for the engine thesis: all three arms are produced by the *same* rollout spine and gap metric,
differing only in a demonstrator + an env wrapper — the `Env`/`ControlPolicy` seams carry the whole study.
The next section swaps the *engine* under that same seam.

## The fidelity rung — re-measured on PhysX (Isaac Sim)

The `Env` seam's claim is that swapping the sim engine re-measures this curve **without touching the policy,
the spine, or the metric**. Tested: `sim/isaac_reach.py` is a rigid-body reach env (`core.rollout.Env`) —
a dynamic sphere in a zero-gravity PhysX scene, actuated by a planar force `gain·clip(a)` with linear
damping `drag/mass`, so its equation of motion is `dv/dt = (gain·clip(a) − drag·v)/mass`: the *identical*
point-mass dynamics, now integrated by PhysX (substeps + implicit damping) instead of explicit Euler.
`sim/isaac_gap.py` imports the **same** `BCPolicy`, `PDExpert`, `Rollout`, and gap metric — only the `Env`
implementation is new. Payload sweep, actuator sag, matched goal seeds all unchanged (3 seeds; opt-in, GPU,
not CI-gated — it needs a ~30 s Isaac kit boot via the opt-in `sim` extra).

| payload (real) | point-mass gap (Euler) | PhysX gap (Isaac) |
|---:|---:|---:|
| +20% | 0.0 ± 0.0 pp | 0.0 ± 0.0 pp |
| +40% | 17.8 ± 5.6 pp | 32.5 ± 4.3 pp |
| +50% | **60.0 ± 4.8 pp** | **62.5 ± 11.5 pp** |
| +60% | 98.0 ± 2.8 pp | 100.0 ± 0.0 pp |

**The finding transfers.** Same margin-then-cliff shape, and the headline **+50% gap is essentially the same
number** across a hand-rolled Euler step and a production PhysX solver (60.0 vs 62.5 pp). The one honest
difference: the cliff is slightly *earlier / sharper* under PhysX (+40% gap 17.8 → 32.5 pp) — the implicit
damping and finite solver substeps make the marginal-payload band a touch harder to reach in-deadline than
Euler's frictionless integration, so the transition payload shifts down a little. That is the expected
direction (more faithful contact/damping → less forgiving), not a contradiction: the mechanism is identical,
the exact transition point sharpens with fidelity — which is precisely why the seam exists.

## Integrity

- **Toy anchor, validated up one rung.** The numpy point mass is a deliberate first rung — fast,
  deterministic, CI-runnable — a *projection* of the sim-to-real mechanism. It is no longer the only
  measurement: the PhysX rung above it (Isaac Sim, `sim/isaac_gap.py`) re-measures the same curve and the
  headline transfers (60.0 → 62.5 pp @ +50%). Both are still simplified reach dynamics, not a real robot
  arm with contact/friction/sensing — the honest ceiling is "the mechanism holds across two solvers", not
  "this is a robotics result".
- **5 seeds, reproducible.** Each BC seed varies net init + demo goals; the run is deterministic (torch +
  env + eval all seeded). The ±sd is the seed spread — small (≤5.6 pp) relative to the transition, so the
  margin-then-cliff shape is not single-init scatter. More seeds would tighten the exact transition payload.
- **The shift is chosen, not fit.** Payload + actuator-sag are mechanical, argued a priori; we sweep the
  whole range rather than cherry-pick one shift, so the reported gap is a point on a disclosed curve.
- **The levers are not tuned to the result.** The DR ranges are the test-shift range itself, fixed a priori
  — not searched for the biggest reduction; DR is reported where it helps (+40%) *and* where it barely moves.
  The adaptive controller uses **observable** proprioception only (`[last_action, Δv]`), never privileged
  `Phys` — so its recovery is a real, deployable compensation, and its residual +60% gap is reported, not
  hidden. Its near-total success is qualified in the caveat above: the toy plant makes online system-ID easy;
  the claim is that adaptive observation *dominates* randomization, not that it dissolves sim-to-real.

## Reproduce

```bash
# point-mass gap + DR lever (fast, CI-gated env)
python -m control.experiment      # prints the table; logs params+metrics to MLflow 'control-policy-gap'

# PhysX fidelity rung (opt-in; needs sim/'s isolated Isaac env + a GPU)
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m control.sim.isaac_gap --seeds 3 --episodes 40
```
