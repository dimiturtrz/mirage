"""hxq.9 — closing the loop the DR lever opened: an *adaptive* policy that senses the payload and beats the
cliff where domain randomization could not.

The DR finding was that a **dynamics-blind** clone (observation `[goal-pos, vel]`) cannot recover the
collapse regime, because it has no cue to tell a heavy payload from a light one. The fix is proprioception,
not more randomization: one step of history — the last action and the velocity change it produced — reveals
the plant gain `k = gain/mass` (from `Δv ≈ (gain·u − drag·v)·dt / mass ≈ k·u·dt`), which an online
controller can invert to restore nominal authority.

Two pieces, both `core.rollout`-native:

- `ProprioEnv` wraps any `core.rollout.Env` and appends `[last_action, Δv]` to its observation, so a
  memoryless policy still sees the one-step system-ID signal (no recurrence needed).
- `AdaptiveExpert` is that online controller — it estimates `k` from `(last_action, Δv)` and commands
  `clip((kp·(goal-pos) − kd·vel) / k̂)`, so a heavier payload (`k̂ < 1`) is met with a proportionally
  larger command. It is the behavior-cloning demonstrator for the adaptive arm; the estimate uses only
  observable proprioception (**not** privileged access to `Phys`), so the cloned student inherits a real,
  deployable compensation. Where `gain/mass` is so small that even the clipped command can't decelerate in
  time, the recovery is capped — an honest physical ceiling, not an estimator failure.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float

from core.rollout import Env, StepResult

_DIM = 2
_VEL0 = 2                # obs[2:4] is velocity in the base [goal-pos, vel] observation
_OBS4 = 4                # base observation width


class ProprioEnv:
    """Wrap a `core.rollout.Env`, appending `[last_action, Δv]` to the observation (base 4-dim → 8-dim).

    The extra channels are the one-step proprioceptive cue an adaptive policy needs to identify the plant
    gain online; at reset they are zero (no history yet), so the first command is the nominal one."""

    def __init__(self, env: Env):
        self._env = env
        self._prev_vel = np.zeros(_DIM)
        self._last_action = np.zeros(_DIM)

    def _augment(self, obs4: Float[np.ndarray, "4"], delta_vel: Float[np.ndarray, "2"]) -> Float[np.ndarray, "8"]:
        return np.concatenate([obs4, self._last_action, delta_vel])

    def reset(self) -> Float[np.ndarray, "8"]:
        obs4 = self._env.reset()
        self._prev_vel = np.asarray(obs4[_VEL0:_OBS4], dtype=float)
        self._last_action = np.zeros(_DIM)
        return self._augment(obs4, np.zeros(_DIM))

    def step(self, action: Float[np.ndarray, "2"]) -> StepResult:
        result = self._env.step(action)
        vel = np.asarray(result.obs[_VEL0:_OBS4], dtype=float)
        delta_vel = vel - self._prev_vel
        self._last_action = np.asarray(action, dtype=float)
        aug = self._augment(result.obs, delta_vel)
        self._prev_vel = vel
        return StepResult(obs=aug, reward=result.reward, done=result.done, success=result.success)


class AdaptiveExpert:
    """Online system-ID PD controller on the augmented observation — the adaptive demonstrator.

    Estimates the scalar plant gain `k = gain/mass` from `‖Δv‖ / (dt·‖last_action‖)` (guarded when no action
    has been applied yet), clips it to a physical band, and inverts it: `a = clip((kp·e − kd·v) / k̂)`. The
    estimate is observable-only; nothing here reads `Phys`. `_K_MIN/_K_MAX/_U_EPS` are fixed estimator guards
    (a physical gain band + a no-command threshold), not per-run knobs — hence class constants, not args."""

    _K_MIN = 0.3
    _K_MAX = 1.5
    _U_EPS = 1e-3

    def __init__(self, dt: float, kp: float = 1.5, kd: float = 1.2, amax: float = 1.0):
        self._dt = dt
        self._kp = kp
        self._kd = kd
        self._amax = amax

    def train(self, _task: str) -> None:
        return None

    def _estimate_gain(self, last_action: Float[np.ndarray, "2"], delta_vel: Float[np.ndarray, "2"]) -> float:
        u_norm = float(np.linalg.norm(last_action))
        if u_norm < self._U_EPS:
            return 1.0                                   # no history / no command yet -> assume nominal plant
        k_hat = float(np.linalg.norm(delta_vel)) / (self._dt * u_norm)
        return float(np.clip(k_hat, self._K_MIN, self._K_MAX))

    def act(self, _state: None, obs: Float[np.ndarray, "8"]) -> Float[np.ndarray, "2"]:
        goal_minus_pos = obs[:_DIM]
        vel = obs[_DIM:_OBS4]
        last_action = obs[_OBS4:_OBS4 + _DIM]
        delta_vel = obs[_OBS4 + _DIM:]
        nominal = self._kp * goal_minus_pos - self._kd * vel
        k_hat = self._estimate_gain(last_action, delta_vel)
        return np.clip(nominal / k_hat, -self._amax, self._amax)
