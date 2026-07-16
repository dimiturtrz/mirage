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
the policy slows before it fails.

### Two levers — randomize over the shift, or sense it

Two arms against the blind baseline. **Domain randomization** clones the same blind expert on randomized
dynamics (payload `~U(1.0, 1.6)`, gain `~U(0.85, 1.0)`). **Adaptive** augments the observation with
`[last_action, Δv]` (a `ProprioEnv`) and clones an online system-ID controller that estimates the plant gain
`k = gain/mass` and inverts it — both matched per seed:

| real payload | nominal gap | DR gap | adaptive gap |
|---:|---:|---:|---:|
| +40% | 17.8 ± 5.6 pp | 7.7 ± 4.4 pp | **0.0 ± 0.0 pp** |
| +50% | **60.0 ± 4.8 pp** | 56.0 ± 1.2 pp | **0.0 ± 0.0 pp** |
| +60% | 98.0 ± 2.8 pp | 93.3 ± 3.3 pp | **18.1 ± 6.9 pp** |

DR only nudges the cliff — the blind policy (obs `[goal-pos, vel]`, no payload cue) can gain marginal-band
robustness but can't beat the actuator limit. The **adaptive policy erases the cliff to +50% (60 → 0 pp)** and
cuts +60% from 98 → 18 pp, from **observable** proprioception only (no privileged `Phys`). The contrast is the
point: *observe* the dynamics shift, don't *average over* it. What remains at +60% is the honest ceiling — a
perfectly-estimated inverse still can't exceed the `amax` clip. (It closes this cleanly because the toy plant
is a clean scalar gain; on a real robot, online ID is much harder — the claim is directional.)

### The fidelity rung — same curve on PhysX (Isaac Sim)

The `Env` seam's payoff: `sim/isaac_reach.py` swaps the numpy point mass for a **PhysX rigid body** (zero
gravity + linear damping `drag/mass` + a planar force → the *same* equation `acc = (gain·a − drag·v)/mass`,
a real solver instead of Euler). `sim/isaac_gap.py` reuses `BCPolicy` / `PDExpert` / `Rollout` / the gap
metric **unchanged** — only the env differs — and the finding transfers:

| payload | point-mass gap | PhysX gap |
|---:|---:|---:|
| +40% | 17.8 ± 5.6 pp | 32.5 ± 4.3 pp |
| +50% | **60.0 ± 4.8 pp** | **62.5 ± 11.5 pp** |
| +60% | 98.0 ± 2.8 pp | 100.0 ± 0.0 pp |

Same margin-then-cliff; the headline +50% gap is essentially unchanged across two solvers. The cliff sits a
touch earlier under PhysX (implicit damping is less forgiving than Euler) — the expected direction, not a
contradiction. Opt-in (GPU + Isaac kit boot, not CI-gated). Full method + integrity →
[`interpretations/control/2026-07-16_policy-gap.md`](../interpretations/control/2026-07-16_policy-gap.md).

## Method — point-mass reach (the fast rung)

- **Task** (`point_mass.py`) — a planar point mass reaches a goal from rest; obs `[goal-pos, vel]`, action a
  clipped acceleration; success = inside `eps` before the horizon. Newtonian step; **sim vs real differ only
  in `Phys`** (payload mass, actuator gain) — the whole sim-to-real story in two mechanical numbers.
- **Expert** (`expert.py`) — an analytic PD controller, the behavior-cloning demonstrator (a `ControlPolicy`).
- **Policy** (`bc.py`) — a torch MLP behavior-cloned on demos (a `ControlPolicy`; torch stays inside the impl,
  the `core` contract is numpy).
- **Adaptive** (`adaptive.py`) — the `ProprioEnv` obs-wrapper (`[last_action, Δv]`) + `AdaptiveExpert`, an
  online system-ID PD controller; the demonstrator for the adaptive arm (observable-only, no privileged `Phys`).
- **Experiment** (`experiment.py`) — train once, roll matched sim/real, sweep payload → the gap curve, over
  N BC seeds, for three arms (nominal / DR / adaptive), logged to MLflow (`control-policy-gap`).

## Integrity

A **numpy point mass is a deliberate first rung** — fast, deterministic, CI-runnable — a *projection* of the
sim-to-real mechanism. The `Env` protocol (`reset`/`step`) is the seam, and it is exercised, not just
claimed: the **PhysX rung** (`sim/isaac_reach.py`) re-measures the same curve at solver fidelity **without
touching the policy, spine, or metric**, and the headline transfers (60.0 → 62.5 pp). Both rungs are still
simplified reach dynamics, not a robot arm with contact/friction/sensing. The shift is chosen (mechanical,
argued a priori) and swept, not cherry-picked; the ± sd is the seed spread, small next to the transition.
The DR ranges are the test-shift range itself, fixed a priori — not searched for the biggest reduction — and
reported where DR helps *and* where it doesn't.

## Run

```bash
python -m control.experiment      # point-mass gap + DR lever; logs to MLflow 'control-policy-gap'
uv run pytest tests/unit/control -q

# PhysX fidelity rung (opt-in; needs sim/'s isolated Isaac env + a GPU):
cd sim && OMNI_KIT_ACCEPT_EULA=YES uv run python isaac_gap.py --seeds 3 --episodes 40
```
