"""A planar point-mass reach task — the minimal continuous-control env with a physically-honest sim-to-real
knob, so the leg's policy gap is measured without an Isaac/MuJoCo install.

State is position + velocity; the action is a clipped acceleration; the observation the policy sees is
`[goal - pos, vel]` (goal-relative, so the same policy generalizes across goals). Dynamics are Newtonian:

    acc = (gain * clip(a) - drag * vel) / mass ;  vel += acc*dt ;  pos += vel*dt

**Sim vs real differ only in `Phys`** — a heavier payload (`mass`) and a weaker actuator (`gain`). That is
the whole sim-to-real story in two numbers, each with a mechanical meaning, no statistical fudge. The env
satisfies `core.rollout.Env` (reset / step -> StepResult), so the rollout spine drives it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from core.rollout import StepResult

_DIM = 2  # planar


@dataclass(frozen=True)
class Phys:
    """The physical parameters that differ between sim and real — a payload mass and an actuator gain, over
    a fixed drag / timestep / action-clip. sim = Phys(); real = Phys(mass=1.5, gain=0.8) is +50% payload,
    -20% actuator authority."""
    mass: float = 1.0
    gain: float = 1.0
    drag: float = 0.1
    dt: float = 0.05
    amax: float = 1.0


@dataclass(frozen=True)
class Task:
    """The reach task geometry — episode horizon, success radius, and the radius the goal is sampled on."""
    horizon: int = 40
    eps: float = 0.15
    goal_radius: float = 1.0


class PointMassReach:
    """Reach a sampled goal from rest; success = inside `eps` of the goal before the horizon expires."""

    def __init__(self, phys: Phys, task: Task, seed: int):
        self._phys = phys
        self._task = task
        self._rng = np.random.default_rng(seed)
        self._pos = np.zeros(_DIM)
        self._vel = np.zeros(_DIM)
        self._goal = np.zeros(_DIM)
        self._t = 0

    def _observe(self) -> Float[np.ndarray, "4"]:
        return np.concatenate([self._goal - self._pos, self._vel])

    def reset(self) -> Float[np.ndarray, "4"]:
        angle = self._rng.uniform(0.0, 2.0 * np.pi)
        self._goal = self._task.goal_radius * np.array([np.cos(angle), np.sin(angle)])
        self._pos = np.zeros(_DIM)
        self._vel = np.zeros(_DIM)
        self._t = 0
        return self._observe()

    def step(self, action: Float[np.ndarray, "2"]) -> StepResult:
        p = self._phys
        a = np.clip(np.asarray(action, dtype=float), -p.amax, p.amax)
        acc = (p.gain * a - p.drag * self._vel) / p.mass
        self._vel = self._vel + acc * p.dt
        self._pos = self._pos + self._vel * p.dt
        self._t += 1
        dist = float(np.linalg.norm(self._pos - self._goal))
        success = dist < self._task.eps
        done = success or self._t >= self._task.horizon
        return StepResult(obs=self._observe(), reward=-dist * p.dt, done=done, success=success)
