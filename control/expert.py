"""The demonstrator — an analytic PD controller that drives the point mass to its goal.

It satisfies `core.policy.ControlPolicy` (train is a no-op; the controller is closed-form), so it rolls
through the same spine as a learned policy and its trajectories are the behavior-cloning targets. Because
the observation is `[goal - pos, vel]`, the control law `a = kp*(goal - pos) - kd*vel` is **linear in the
observation** — until the action clip saturates, which is exactly the nonlinearity a linear clone cannot
reproduce and an MLP clone can.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float

_OBS_SPLIT = 2  # obs = [goal - pos (2), vel (2)]


class PDExpert:
    """Closed-form PD controller to the goal; the behavior-cloning demonstrator."""

    def __init__(self, kp: float = 1.5, kd: float = 1.2, amax: float = 1.0):
        self._kp = kp
        self._kd = kd
        self._amax = amax

    def train(self, _task: str) -> Any:
        return None

    def act(self, _state: Any, obs: Float[np.ndarray, "4"]) -> Float[np.ndarray, "2"]:
        goal_minus_pos = obs[:_OBS_SPLIT]
        vel = obs[_OBS_SPLIT:]
        return np.clip(self._kp * goal_minus_pos - self._kd * vel, -self._amax, self._amax)
