# Three imitation paradigms, one spine — does action-chunking change the sim-to-real policy gap?

**Date:** 2026-07-18 · **Epic:** hxq (child hxq.2) · **Code:** `control/{bc,act,diffusion_policy}.py` · **Run:** `python -m control.policy_compare`

## Question

The control leg fills out its model family: behavior cloning (reactive MLP), **ACT** (an action-chunking
transformer, Zhao et al. 2023), and a **diffusion policy** (conditional DDPM over action chunks, Chi et al.
2023) — the three imitation paradigms a modern VLA stack is built from. They all satisfy the same
`core.policy` `(train, act)` contract and roll through the same `core.rollout` spine, so the question the
eval discipline asks is direct: **cloned from the same demos and measured by the same gap metric, does the
learning paradigm move the sim-to-real number?** (Filling the family also gives the edge-VLA step, hxq.4, a
real transformer / iterative-denoiser graph to distill+quantize — an MLP has nothing to distill.)

## Method — same demos, same shift, three policies

- **Shared demos.** All three clone the *same* PD expert from *nominal-sim* demos (200 episodes). BC regresses
  the per-step action; ACT and diffusion predict the next **K=8** actions (a chunk) per observation, from the
  shared `Demos.chunks` collector (tail-padded by holding the last command).
- **BC** — a 2-layer MLP obs→action (the reactive baseline; unchanged from the gap study).
- **ACT** — a transformer decoder attends K learned query tokens over the encoded observation and emits an
  action chunk; executed **receding-horizon** (predict a chunk, take its first action, re-predict next step).
- **Diffusion policy** — a conditional ε-network denoises a Gaussian action chunk over 50 DDPM steps,
  conditioned on the observation; sampled fresh each step, first action executed (diffusion policy's own
  deployment mode). The forward/reverse coefficient math is the pure, unit-tested `DiffusionSchedule`.
- **Eval.** Each policy is rolled over 200 matched-seed episodes in nominal sim and in the **headline +50%
  payload / −10% actuator** real (the shift the BC gap curve identifies as the cliff). The gap carries a **95%
  paired-bootstrap CI over the matched episodes**, computed through the promoted `core.metrics.boot_delta_ci` —
  the *same* primitive that brackets the perception AU-PRO gap (eval-spine promotion, xut.1). This brackets
  eval-episode noise; it is a **single training seed** (net init), so the init-variance hardening is separate
  (house rule: quick-signal first; the multi-seed init hardening is filed, 7wz).

## Result (single seed, +50% payload)

| policy | sim success | real success | sim-to-real gap [95% paired-boot CI] |
|---|---:|---:|---:|
| behavior cloning | 100.0% | 44.5% | **55.5 pp** [48.5, 63.0] |
| ACT (chunk transformer) | 100.0% | 25.5% | 74.5 pp [68.0, 80.5] |
| diffusion policy | 100.0% | 27.0% | 73.0 pp [67.0, 79.5] |

BC's single-seed +50% real (44.5%) sits inside the 5-seed gap study's 40.0 ± 4.8% — the paradigms are
compared against a consistent baseline. **The reactive CI [48.5, 63.0] does not overlap either chunked CI
([68.0, 80.5], [67.0, 79.5])**, so over the 200 matched eval episodes the chunked-carries-more-gap ordering is
a real separation, not episode noise.

## Interpretation — chunking buys temporal consistency, and pays for it in sim-to-real robustness

**All three learn the task in-domain** (100% sim): the contract and spine carry a transformer and an iterative
denoiser as transparently as the MLP — that is the packaging result, three paradigms on one seam.

**But the chunked paradigms carry a *larger* sim-to-real gap** (74.5 / 73.0 pp vs BC's 55.5 pp). This is not
noise in the paradigm's favor and it is worth stating plainly: on this dynamics shift, action-chunking is
*more* sim-optimistic than reactive cloning. The mechanism is the same tradeoff that motivates chunking in the
first place. ACT and diffusion predict a short-horizon *trajectory* of actions — intent shaped by an implicit
model of how the plant will respond — and that model was learned on nominal dynamics. Under a +50% payload the
plant responds differently, so the chunk-shaped first action is calibrated to a response that no longer holds.
BC's reactive map has no trajectory model to be wrong: it reads the current `[goal-pos, vel]` and reacts, and
that single-step feedback is exactly what absorbs the dynamics mismatch (the same reason the gap study found
closed-loop feedback erases the gap up to +30%). **Chunking trades reactivity for temporal consistency; under a
dynamics shift, reactivity is the property that transfers.**

This is the leg's thesis working as intended: the sim-to-real eval discipline *measures* a paradigm tradeoff
that an in-domain success number (all 100%) completely hides. A leaderboard would call these three policies
identical; the gap metric separates them and names the cost of chunking under shift.

## Integrity

- **CI-separated over episodes, single training seed.** The paired-bootstrap CIs (via `core.metrics`) show the
  chunked>reactive ordering is a real separation over the 200 matched eval episodes — more than directional. But
  that CI brackets *eval-episode* noise, not *net-init* noise: this is one training seed, so the init-variance
  hardening (multi-seed, matching `experiment.py`) is the remaining rigor and is filed (7wz).
- **The toy task does not flatter chunking.** Point-mass reach is unimodal and low-dimensional — it exercises
  none of what ACT/diffusion were built for (multimodal human demos, high-dim visuomotor input). So this is
  *not* evidence that chunking is a worse paradigm in general; it is evidence that on a reactive control task
  under a dynamics shift, chunking's trajectory commitment costs sim-to-real robustness. Reported as the honest
  finding, not spun — the models earn their place as the **distillable graph substrate for edge-VLA (hxq.4)**,
  and the gap they carry is a measured property, not a defect to hide.
- **Faithful reductions, argued.** ACT drops the CVAE latent (our demonstrator is deterministic — a latent
  would model nothing) and uses receding-horizon execution instead of temporal ensembling (which needs
  per-episode buffer state the stateless `(state, obs)→action` contract doesn't carry). Both are documented in
  the module docstrings, not silent. The DDPM schedule is the standard linear-β chain, unit-tested in isolation.

## Reproduce

```bash
python -m control.policy_compare      # trains bc/act/diffusion on shared demos; logs to MLflow 'control-policy-gap'
uv run pytest tests/unit/control -q
```

See also `interpretations/control/2026-07-16_policy-gap.md` (the BC gap curve, DR/adaptive levers, PhysX rung).
