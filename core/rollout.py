"""The rollout spine — the control-leg analog of `harness.aggregate`: it drives a `ControlPolicy` through
an environment and records the episode as a `Trajectory`, then reduces trajectories to the headline
**sim-to-real policy gap**.

Where the perception harness scores a `(fit, score)` method into `ScoreArrays` → AU-PRO/AUROC, this rolls
a `(train, act)` policy into a `Trajectory` → success-rate / return, and `gap` subtracts the two rollout
sets (sim vs real) into the one number the control leg exists to measure. Pure typing + numpy: no sim
engine, no torch — an `Env` is anything with `reset`/`step`, so the spine is exercised here by a stub and
later by Isaac Lab / MuJoCo (bead hxq.3) without touching this file.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Shaped

from core.policy import ControlPolicy, StateT, Trajectory


class EvalPlan(NamedTuple):
    """How many seed-matched episodes to roll, from which seed, over how long a horizon — the eval/collection
    spec the rollout set and the demo collector share (matched seeds isolate the shift from goal luck)."""
    episodes: int
    seed0: int
    max_steps: int


class StepResult(NamedTuple):
    """One environment transition (the tuple `Env.step` returns; the spine stacks these into a Trajectory)."""
    obs: np.ndarray      # (...)   next observation
    reward: float        # scalar step reward
    done: bool           # episode-terminated flag
    success: bool        # task-success flag


@runtime_checkable
class Env(Protocol):
    """The minimal environment contract the spine drives — `reset` to a first observation, `step` an action
    to a `StepResult`. Isaac Lab / MuJoCo wrappers and the test's `CountdownEnv` alike satisfy it."""

    def reset(self) -> Shaped[np.ndarray, "*obs"]: ...

    def step(self, action: Shaped[np.ndarray, "*act"]) -> StepResult: ...


class Rollout:
    """Drive a policy through an env into a Trajectory, then reduce trajectories to the policy gap."""

    @staticmethod
    def roll(policy: ControlPolicy[StateT], state: StateT, env: Env, max_steps: int) -> Trajectory:
        """Roll a policy already trained to `state` (call `policy.train(task)` once, then roll many episodes —
        the control analog of fit-once-score-many; training inside the loop would re-fit per episode)."""
        obs = env.reset()
        obs_log: list[Shaped[np.ndarray, "*obs"]] = []
        act_log: list[Shaped[np.ndarray, "*act"]] = []
        rew_log: list[float] = []
        done_log: list[bool] = []
        ok_log: list[bool] = []
        for _ in range(max_steps):
            action = policy.act(state, obs)
            step = env.step(action)
            obs_log.append(obs)
            act_log.append(action)
            rew_log.append(step.reward)
            done_log.append(step.done)
            ok_log.append(step.success)
            obs = step.obs
            if step.done:
                break
        return Trajectory(
            obs=np.asarray(obs_log),
            actions=np.asarray(act_log),
            rewards=np.asarray(rew_log, dtype=float),
            dones=np.asarray(done_log, dtype=bool),
            success=np.asarray(ok_log, dtype=bool),
        )

    @staticmethod
    def rollset(policy: ControlPolicy[StateT], state: StateT, make_env: Callable[[int], Env],
                plan: EvalPlan) -> list[Trajectory]:
        """Roll `plan.episodes` seed-matched episodes (`plan.seed0 + i`) of a trained policy — the eval set the
        metrics reduce. Matched seeds across a sim set and a real set isolate the dynamics shift from goal luck."""
        return [Rollout.roll(policy, state, make_env(plan.seed0 + i), plan.max_steps)
                for i in range(plan.episodes)]

    @staticmethod
    def success_rate(trajs: list[Trajectory]) -> float:
        return float(np.mean([t.success.any() for t in trajs])) if trajs else 0.0

    @staticmethod
    def mean_return(trajs: list[Trajectory]) -> float:
        return float(np.mean([t.rewards.sum() for t in trajs])) if trajs else 0.0

    @staticmethod
    def gap(sim: list[Trajectory], real: list[Trajectory]) -> float:
        """The sim-to-real policy gap (pp): sim success-rate minus real. Positive = sim-optimistic."""
        return Rollout.success_rate(sim) - Rollout.success_rate(real)
