# control — the sim-to-real policy gap

The control/VLA leg: the engine's sim-to-real *eval* discipline applied to a second axis — **action**, not
perception. It shares `core` with `surfscan` (import-linter enforces the two legs are independent), speaking
the `core.policy` `(train, act)` contract and the `core.rollout` spine — the control analog of perception's
`(fit, score)` / `harness.aggregate`.

## The number

A policy is **behavior-cloned in nominal sim only**, then rolled in sim and in a dynamics-shifted **real**
(unmodeled payload + −10% actuator gain) over matched goal seeds. The headline is the **sim-to-real policy
gap** = success(sim) − success(real), swept over payload, 5 BC seeds (mean ± sd):

| real payload | task success | sim-to-real gap |
|---:|---:|---:|
| +10 … +30% | 100.0 ± 0.0% | 0.0 ± 0.0 pp |
| +40% | 82.2 ± 5.6% | 17.8 ± 5.6 pp |
| +50% | 40.0 ± 4.8% | **60.0 ± 4.8 pp** |
| +60% | 2.0 ± 2.8% | 98.0 ± 2.8 pp |

**A threshold, not a slope.** Closed-loop feedback absorbs ≤ +30% payload (zero gap, zero variance across
seeds); past ~+45% the actuator can't overcome the added inertia within the horizon and success collapses.
An early-warning precedes the cliff: mean steps-to-goal climbs (36.5 → 38.6) while success is still 100% —
the policy slows before it fails. Full method + integrity →
[`interpretations/control/2026-07-16_policy-gap.md`](../interpretations/control/2026-07-16_policy-gap.md).

## Method — point-mass reach, no Isaac

- **Task** (`point_mass.py`) — a planar point mass reaches a goal from rest; obs `[goal-pos, vel]`, action a
  clipped acceleration; success = inside `eps` before the horizon. Newtonian step; **sim vs real differ only
  in `Phys`** (payload mass, actuator gain) — the whole sim-to-real story in two mechanical numbers.
- **Expert** (`expert.py`) — an analytic PD controller, the behavior-cloning demonstrator (a `ControlPolicy`).
- **Policy** (`bc.py`) — a torch MLP behavior-cloned on nominal-sim demos (a `ControlPolicy`; torch stays
  inside the impl, the `core` contract is numpy).
- **Experiment** (`experiment.py`) — train once, roll matched sim/real, sweep payload → the gap curve, over
  N BC seeds, logged to MLflow (`control-policy-gap`).

## Integrity

A **numpy point mass is a deliberate first rung** — fast, deterministic, CI-runnable — a *projection* of the
sim-to-real mechanism, not a high-fidelity robot. The `Env` protocol (`reset`/`step`) is the seam: an **Isaac
Lab / MuJoCo** env re-measures this exact curve at higher fidelity **without touching the policy, spine, or
metric**. The shift is chosen (mechanical, argued a priori) and swept, not cherry-picked; the ± sd is the
5-seed spread (≤ 5.6 pp), small next to the transition.

## Run

```bash
python -m control.experiment      # prints the gap curve; logs params+metrics to MLflow 'control-policy-gap'
uv run pytest tests/unit/control -q
```
