"""The policy contract the control harness rolls out — the control-leg analog of `method.py`.

Where `method.py` names the perception contract (a `(fit, score)` pair scored per sample), this names
the control contract: every policy, whatever its paradigm, is a `(train_fn, act_fn)` pair the harness
speaks to:

    train_fn(task)          -> state           # build/load the policy for one task
    act_fn(state, obs)      -> action          # one step; the harness rolls act over an episode

The harness rolls `act` in the sim loop and records the episode as a `Trajectory` — the control-side
analog of `ScoreArrays`. The mapping to `method.py` is deliberate and exact:

    ScoreArrays (per-sample)  ->  Trajectory (per-timestep)
    fit(cat) -> state         ->  train(task) -> state
    score(state, cat)         ->  act(state, obs)   (harness rolls act -> Trajectory)
    Method(fit, score)        ->  Policy(train, act)
    AnomalyMethod Protocol    ->  ControlPolicy Protocol

Because behaviour-cloning, RL, and VLA-distill policies all satisfy it, one rollout spine scores them
identically. `Trajectory` names its fields so the harness reads them by name, not by tuple position.
Pure typing + numpy: no torch/sim/surfscan import — `core` stays the independent kernel (import-linter).
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Shaped


class Trajectory(NamedTuple):
    """One rolled-out episode (all aligned on axis 0 = T timesteps; the harness reads them by name)."""
    obs: np.ndarray      # (T, ...)  observation fed to the policy each step
    actions: np.ndarray  # (T, ...)  action the policy emitted           -> sim-to-real policy gap
    rewards: np.ndarray  # (T,)      per-step reward                     -> return
    dones: np.ndarray    # (T,)      bool episode-terminated flag
    success: np.ndarray  # (T,)      bool task-success flag (sticky once true) -> success rate


TrainFn = Callable[[str], Any]                                         # train(task) -> state
ActFn = Callable[[Any, Shaped[np.ndarray, "*obs"]], Shaped[np.ndarray, "*act"]]  # act(state, obs) -> action


class Policy(NamedTuple):
    """The (train, act) pair the harness rolls out — the closure form of `ControlPolicy` (for ad-hoc
    or orchestrated policies that don't warrant a class; e.g. a scripted or distilled per-task policy)."""
    train: TrainFn
    act: ActFn


@runtime_checkable
class ControlPolicy(Protocol):
    """The minimal contract the rollout harness speaks to — a configured policy object. Deliberately just
    `train`/`act`: everything else (env construction, a Trainer, the mlflow `run_name`) is composed by
    HAS-A, never bolted on as protocol hooks. Both the `Policy` closure-pair and the per-paradigm policy
    classes (BCPolicy, distilled-VLA, …) satisfy it, so the harness reads `.train`/`.act` without caring
    which — the exact shape `AnomalyMethod` has on the perception side."""

    def train(self, task: str) -> Any: ...

    def act(self, state: Any, obs: Shaped[np.ndarray, "*obs"]) -> Shaped[np.ndarray, "*act"]: ...
