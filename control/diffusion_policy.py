"""Diffusion policy (Chi et al. 2023) — model the action chunk as the output of a conditional denoising
process instead of a direct regression.

Where BC and ACT regress the expert action(s) directly, a diffusion policy learns to *denoise*: train a
network to predict the noise added to an action chunk at a random level, conditioned on the observation; at
inference, start from Gaussian noise and run the reverse chain to a clean chunk. On a unimodal reach the
regressors already suffice — the reason it earns its place here is the deployment story: an iterative
denoiser is the archetype the edge-VLA step (hxq.4) distills (few-step / consistency distillation) and
quantizes, a graph a one-shot MLP can't stand in for.

`DiffusionPolicy` satisfies `core.policy.ControlPolicy`: `train` fits the ε-network on `Demos.chunks`; `act`
samples a chunk by running the reverse chain and executes its first action (receding horizon — diffusion
policy's own deployment mode). The forward/reverse coefficient math is the pure `DiffusionSchedule`; torch
supplies only the learned ε estimate, so the schedule stays unit-tested off the GPU.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from control.demos import Demos
from control.diffusion_schedule import DiffusionSchedule
from core.rollout import EvalPlan


@dataclass(frozen=True)
class DiffusionConfig:
    """Diffusion-policy knobs — demo volume, ε-net width, denoising steps, chunk length, fit schedule.

    `steps=50` is a standard small-DDPM chain (enough reverse resolution, cheap to sample); `hidden=128` is
    wider than the BC MLP because the ε-net's input is the whole flattened chunk plus the noise level, not a
    single observation. `chunk=8` matches ACT so the chunked paradigms are compared like-for-like."""
    n_demo_episodes: int = 200
    hidden: int = 128
    steps: int = 50
    chunk: int = 8
    epochs: int = 300
    lr: float = 1e-3
    max_steps: int = 40
    seed: int = 0


class EpsMLP(nn.Module):
    """Noise estimator ε_θ(obs, noised-chunk, t) — three hidden layers over the conditioned, flattened chunk."""

    def __init__(self, obs_dim: int, chunk_flat: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + chunk_flat + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, chunk_flat),
        )

    def forward(self, obs: Float[Tensor, "b obs"], x_t: Float[Tensor, "b flat"],
                t_norm: Float[Tensor, "b 1"]) -> Float[Tensor, "b flat"]:
        return self.net(torch.cat([obs, x_t, t_norm], dim=1))


class DiffusionPolicy:
    """Fit a conditional ε-network on expert chunks; sample a chunk by reverse diffusion, execute action 0."""

    def __init__(self, sim_env_factory: Callable[[int], Any], expert: Any, cfg: DiffusionConfig):
        self._make_sim = sim_env_factory
        self._expert = expert
        self._cfg = cfg
        self._sched = DiffusionSchedule(cfg.steps)
        self._act_dim = 0

    def train(self, task: str) -> Any:
        torch.manual_seed(self._cfg.seed)
        plan = EvalPlan(self._cfg.n_demo_episodes, self._cfg.seed, self._cfg.max_steps)
        obs, chunks = Demos.chunks(Demos.rollouts(self._expert, task, self._make_sim, plan), self._cfg.chunk)
        self._act_dim = chunks.shape[2]
        flat = chunks.reshape(chunks.shape[0], -1)
        net = EpsMLP(obs.shape[1], flat.shape[1], self._cfg.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self._cfg.lr)
        ob = torch.as_tensor(obs, dtype=torch.float32)
        y0 = torch.as_tensor(flat, dtype=torch.float32)
        sqrt_ab = torch.as_tensor(np.sqrt(self._sched.alphas_cumprod), dtype=torch.float32)
        sqrt_1mab = torch.as_tensor(np.sqrt(1.0 - self._sched.alphas_cumprod), dtype=torch.float32)
        for _ in range(self._cfg.epochs):
            t = torch.randint(0, self._cfg.steps, (y0.shape[0],))
            noise = torch.randn_like(y0)
            x_t = sqrt_ab[t].unsqueeze(1) * y0 + sqrt_1mab[t].unsqueeze(1) * noise
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(ob, x_t, (t.float() / self._cfg.steps).unsqueeze(1)), noise)
            loss.backward()
            opt.step()
        return net

    def _sample(self, net: Any, obs: Float[np.ndarray, "obs"], rng: np.random.Generator) -> Float[np.ndarray, "flat"]:
        ob = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        x = rng.standard_normal(self._cfg.chunk * self._act_dim).astype(np.float32)
        for t in reversed(range(self._cfg.steps)):
            t_norm = torch.full((1, 1), t / self._cfg.steps, dtype=torch.float32)
            with torch.no_grad():
                eps = net(ob, torch.as_tensor(x, dtype=torch.float32).unsqueeze(0), t_norm)[0].numpy()
            x = self._sched.posterior_mean(x, eps, t)
            if t > 0:
                x = x + np.sqrt(self._sched.betas[t]) * rng.standard_normal(x.shape).astype(np.float32)
        return x

    def act(self, state: Any, obs: Float[np.ndarray, "obs"]) -> Float[np.ndarray, "act"]:
        rng = np.random.default_rng(self._cfg.seed)
        chunk = self._sample(state, obs, rng).reshape(self._cfg.chunk, self._act_dim)
        return chunk[0]                                             # first action of the sampled chunk
