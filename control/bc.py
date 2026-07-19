"""Behavior cloning — fit an MLP to map observations to the PD expert's actions, collected in the **nominal
sim** only.

This is the policy whose sim-to-real gap the leg measures: it never sees the shifted-real dynamics, so
rolling it in real reveals how much a sim-trained policy degrades. `BCPolicy` satisfies
`core.policy.ControlPolicy` — `train` collects expert demos through the rollout spine and fits the net;
`act` runs the net. torch lives here, inside the policy implementation; the `core` contract stays numpy.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, override

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from control.demos import Demos
from core.policy import ControlPolicy, StateT
from core.rollout import Env, EvalPlan


@dataclass(frozen=True)
class BCConfig:
    """Behavior-cloning knobs — demo volume, net width, and the fit schedule."""
    n_demo_episodes: int = 200
    hidden: int = 64
    epochs: int = 300
    lr: float = 1e-3
    max_steps: int = 40
    seed: int = 0


class MLP(nn.Module):
    """Two-hidden-layer MLP regressor obs -> action."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    @override
    def forward(self, x: Float[Tensor, "b in"]) -> Float[Tensor, "b out"]:
        return self.net(x)


class BCPolicy(Generic[StateT]):
    """Clone the PD expert from nominal-sim demos; the sim-trained policy the gap is measured on.

    Generic in the *demonstrator's* state type (`StateT`): the clone is agnostic to what its expert carries
    between train and act (the PD experts carry nothing), while its own state is the fitted `MLP`."""

    def __init__(self, sim_env_factory: Callable[[int], Env], expert: ControlPolicy[StateT], cfg: BCConfig):
        self._make_sim = sim_env_factory
        self._expert = expert
        self._cfg = cfg

    def _collect(self, task: str) -> tuple[Float[np.ndarray, "n 4"], Float[np.ndarray, "n 2"]]:
        plan = EvalPlan(self._cfg.n_demo_episodes, self._cfg.seed, self._cfg.max_steps)
        return Demos.flat(Demos.rollouts(self._expert, task, self._make_sim, plan))

    def train(self, task: str) -> MLP:
        torch.manual_seed(self._cfg.seed)
        obs, act = self._collect(task)
        net = MLP(obs.shape[1], self._cfg.hidden, act.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self._cfg.lr)
        xb = torch.as_tensor(obs, dtype=torch.float32)
        yb = torch.as_tensor(act, dtype=torch.float32)
        for _ in range(self._cfg.epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb), yb)
            loss.backward()
            opt.step()
        return net

    def act(self, state: nn.Module, obs: Float[np.ndarray, "4"]) -> Float[np.ndarray, "2"]:
        net = state
        with torch.no_grad():
            return net(torch.as_tensor(obs, dtype=torch.float32)).numpy()
