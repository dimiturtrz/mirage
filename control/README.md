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

The `Env` seam's payoff: `control/sim/isaac_reach.py` swaps the numpy point mass for a **PhysX rigid body**
(zero gravity + linear damping `drag/mass` + a planar force → the *same* equation `acc = (gain·a − drag·v)/mass`,
a real solver instead of Euler). `control/sim/isaac_gap.py` reuses `BCPolicy` / `PDExpert` / `Rollout` / the gap
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

### The paradigm axis — reactive vs chunked (BC / ACT / diffusion)

The leg speaks all three imitation paradigms through the one `(train, act)` contract: reactive behavior
cloning, an **ACT** action-chunking transformer, and a **diffusion policy** (conditional DDPM over action
chunks). Cloned from the *same* nominal demos and scored by the *same* gap metric at the headline +50% shift
(single seed):

| policy | sim success | real success | sim-to-real gap [95% CI] |
|---|---:|---:|---:|
| behavior cloning | 100.0% | 44.5% | **55.5 pp** [48.5, 63.0] |
| ACT (chunk transformer) | 100.0% | 25.5% | 74.5 pp [68.0, 80.5] |
| diffusion policy | 100.0% | 27.0% | 73.0 pp [67.0, 79.5] |

All three learn the task in-domain; the **chunked** paradigms carry a *larger* sim-to-real gap. Action-chunking
commits to trajectory-shaped intent calibrated to nominal dynamics, where BC's single-step feedback is exactly
what absorbs the payload shift — the eval discipline names a paradigm tradeoff that the 100% in-domain success
completely hides. The gap is bracketed by a **paired bootstrap over the matched episodes through the shared
`core.metrics` substrate** — the *same* primitive that brackets the perception AU-PRO gap (promoted into `core`
so both legs score through one honest-uncertainty story). The reactive and chunked CIs do not overlap, so the
ordering is a real separation over the 200 eval episodes, not scatter — over a single training seed still
(multi-seed init hardening filed). Method + mechanism →
[`interpretations/control/2026-07-18_paradigm-gap.md`](../interpretations/control/2026-07-18_paradigm-gap.md).

### Edge-VLA — distill the heaviest graph to the edge, quantize, measure the cost

The differentiator: a policy that runs real-time on-device, demonstrable **sim-in-the-loop without hardware**.
The diffusion policy is the worst-case edge graph — 50 denoising steps (50 network evals) per action. Distill
it into a **1-step** student (an MLP regressing the teacher's sampled chunk) and **int8-quantize**:

| policy | steps | params (k) | size (kB) | op-class | sim | real | gap (pp) |
|---|---:|---:|---:|---|---:|---:|---:|
| diffusion (teacher) | 50 | 37.9 | 151.1 | iterative_dense | 100.0% | 27.0% | 73.0 |
| distilled (fp32) | 1 | 19.2 | 77.8 | dense | 100.0% | 23.0% | 77.0 |
| distilled (int8) | 1 | 19.2 | 24.2 | dense | 100.0% | 22.5% | 77.5 |

**Distillation cuts inference 50× (50 evals → 1) and halves the graph for +4 pp of gap; int8 shrinks it 3.2×
(78 → 24 kB) nearly free (+0.5 pp).** The product is a **24 kB single-pass dense-op-class policy** — edge-fit.
The op-class verdict repeats the deploy explorer's finding: the iterative diffusion teacher pays latency × 50
(edge-prohibitive at 30 fps), ACT's **attention** doesn't compile to a fixed NPU (the same wall the Point-MAE
detector hit), and the **distilled+quantized dense student ships on any accelerator, fixed NPUs included**. So:
distill an attention/iterative policy to a dense single-pass student, quantize, deploy. Projection (params /
size / steps), not measured FPS — the hardware bench is the deploy leg (bead 0sq). Method + verdict →
[`interpretations/control/2026-07-18_edge-vla.md`](../interpretations/control/2026-07-18_edge-vla.md).

## Method — point-mass reach (the fast rung)

- **Task** (`point_mass.py`) — a planar point mass reaches a goal from rest; obs `[goal-pos, vel]`, action a
  clipped acceleration; success = inside `eps` before the horizon. Newtonian step; **sim vs real differ only
  in `Phys`** (payload mass, actuator gain) — the whole sim-to-real story in two mechanical numbers.
- **Expert** (`expert.py`) — an analytic PD controller, the behavior-cloning demonstrator (a `ControlPolicy`).
- **Policy family** — three `ControlPolicy` impls cloned from shared demos (`demos.py`): `bc.py` (reactive
  MLP), `act.py` (ACT action-chunking transformer), `diffusion_policy.py` (conditional DDPM over action chunks,
  pure schedule in `diffusion_schedule.py`). torch stays inside each impl; the `core` contract is numpy.
- **Adaptive** (`adaptive.py`) — the `ProprioEnv` obs-wrapper (`[last_action, Δv]`) + `AdaptiveExpert`, an
  online system-ID PD controller; the demonstrator for the adaptive arm (observable-only, no privileged `Phys`).
- **Experiment** (`experiment.py`) — train once, roll matched sim/real, sweep payload → the gap curve, over
  N BC seeds, for three arms (nominal / DR / adaptive), logged to MLflow (`control-policy-gap`).
- **Paradigm compare** (`policy_compare.py`) — clone bc / act / diffusion on the same demos, roll matched
  sim/real at the headline shift → the per-paradigm gap, logged to MLflow (`control-policy-gap`).
- **Edge-VLA** (`edge.py`) — distill the diffusion policy (50-step) into a 1-step `DistilledPolicy`,
  int8-quantize it, and report footprint (params/size/steps) vs task success + the op-class edge verdict.

## Integrity

A **numpy point mass is a deliberate first rung** — fast, deterministic, CI-runnable — a *projection* of the
sim-to-real mechanism. The `Env` protocol (`reset`/`step`) is the seam, and it is exercised, not just
claimed: the **PhysX rung** (`control/sim/isaac_reach.py`) re-measures the same curve at solver fidelity **without
touching the policy, spine, or metric**, and the headline transfers (60.0 → 62.5 pp). Both rungs are still
simplified reach dynamics, not a robot arm with contact/friction/sensing. The shift is chosen (mechanical,
argued a priori) and swept, not cherry-picked; the ± sd is the seed spread, small next to the transition.
The DR ranges are the test-shift range itself, fixed a priori — not searched for the biggest reduction — and
reported where DR helps *and* where it doesn't.

## Run

```bash
python -m control.experiment      # point-mass gap + DR/adaptive levers; logs to MLflow 'control-policy-gap'
python -m control.policy_compare  # bc/act/diffusion paradigm gap on the shared spine
python -m control.edge            # edge-VLA: distill (50-step -> 1) + int8 quantize, footprint vs quality
uv run pytest tests/unit/control -q

# PhysX fidelity rung (opt-in; needs the `sim` extra + a GPU):
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m control.sim.isaac_gap --seeds 3 --episodes 40
```
