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
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from core.rollout import Rollout


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

    def forward(self, x: Float[Tensor, "b in"]) -> Float[Tensor, "b out"]:
        return self.net(x)


class BCPolicy:
    """Clone the PD expert from nominal-sim demos; the sim-trained policy the gap is measured on."""

    def __init__(self, sim_env_factory: Callable[[int], Any], expert: Any, cfg: BCConfig):
        self._make_sim = sim_env_factory
        self._expert = expert
        self._cfg = cfg

    def _collect(self, task: str) -> tuple[Float[np.ndarray, "n 4"], Float[np.ndarray, "n 2"]]:
        obs_all, act_all = [], []
        expert_state = self._expert.train(task)
        for i in range(self._cfg.n_demo_episodes):
            env = self._make_sim(self._cfg.seed + i)
            traj = Rollout.roll(self._expert, expert_state, env, self._cfg.max_steps)
            obs_all.append(traj.obs)
            act_all.append(traj.actions)
        return np.concatenate(obs_all), np.concatenate(act_all)

    def train(self, task: str) -> Any:
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

    def act(self, state: Any, obs: Float[np.ndarray, "4"]) -> Float[np.ndarray, "2"]:
        net = state
        with torch.no_grad():
            return net(torch.as_tensor(obs, dtype=torch.float32)).numpy()
