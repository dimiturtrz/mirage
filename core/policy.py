"""The policy contract the control harness rolls out — the control-leg analog of `method.py`.

Where `method.py` names the perception contract (a `(fit, score)` pair scored per sample), this names
the control contract: every policy, whatever its paradigm, is a `(train, act)` pair the harness speaks to:

    train(task)          -> state           # build/load the policy for one task
    act(state, obs)      -> action          # one step; the harness rolls act over an episode

The harness rolls `act` in the sim loop and records the episode as a `Trajectory` — the control-side
analog of `ScoreArrays`. The mapping to `method.py` is deliberate and exact:

    ScoreArrays (per-sample)  ->  Trajectory (per-timestep)
    fit(cat) -> state         ->  train(task) -> state
    score(state, cat)         ->  act(state, obs)   (harness rolls act -> Trajectory)
    AnomalyMethod Protocol    ->  ControlPolicy Protocol

Because behaviour-cloning, RL, and VLA-distill policies all satisfy it, one rollout spine scores them
identically. `Trajectory` names its fields so the harness reads them by name, not by tuple position.
(`method.py` also ships a `Method` closure-pair for ad-hoc methods because the triad uses it; the analogous
`Policy(train, act)` closure is added here when a closure-based policy first needs it — none does yet.)
Pure typing + numpy: no torch/sim/surfscan import — `core` stays the independent kernel (import-linter).
"""
from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Shaped


class Trajectory(NamedTuple):
    """One rolled-out episode (all aligned on axis 0 = T timesteps; the harness reads them by name)."""
    obs: np.ndarray      # (T, ...)  observation fed to the policy each step
    actions: np.ndarray  # (T, ...)  action the policy emitted           -> sim-to-real policy gap
    rewards: np.ndarray  # (T,)      per-step reward                     -> return
    dones: np.ndarray    # (T,)      bool episode-terminated flag        -> steps-to-goal
    success: np.ndarray  # (T,)      bool task-success flag (sticky once true) -> success rate


@runtime_checkable
class ControlPolicy(Protocol):
    """The minimal contract the rollout harness speaks to — a configured policy object. Deliberately just
    `train`/`act`: everything else (env construction, a Trainer, the mlflow `run_name`) is composed by
    HAS-A, never bolted on as protocol hooks. The per-paradigm policy classes (PDExpert, BCPolicy,
    distilled-VLA, …) satisfy it, so the harness reads `.train`/`.act` without caring which — the exact
    shape `AnomalyMethod` has on the perception side."""

    def train(self, _task: str) -> Any: ...

    def act(self, _state: Any, _obs: Shaped[np.ndarray, "*obs"]) -> Shaped[np.ndarray, "*act"]: ...
