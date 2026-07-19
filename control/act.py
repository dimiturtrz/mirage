"""ACT — an action-chunking transformer policy (Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation
with Low-Cost Hardware"): predict the next K actions in one shot instead of one step at a time.

The chunk is the point. A transformer decoder attends K learned query tokens over the encoded observation and
emits a whole action sub-sequence, so the policy commits to short-horizon *intent* rather than re-deciding
every step — the mechanism that damps compounding single-step cloning error. `ACTPolicy` satisfies
`core.policy.ControlPolicy`: `train` fits the chunk transformer on `Demos.chunks`, `act` predicts a chunk and
executes its first action (receding horizon — re-predict each step).

Two deliberate reductions from the paper, each argued rather than hidden:
- **No CVAE latent.** ACT's variational encoder models multimodal *human* demonstrations; our demonstrator is
  a deterministic controller, so a latent would train to a point mass and add machinery that carries no
  signal (house rule: derive with an argument, don't add unused capacity).
- **Receding-horizon first-action, not temporal ensembling.** Ensembling averages overlapping chunk
  predictions across timesteps, which needs per-episode buffer state the stateless `(state, obs) -> action`
  contract doesn't carry; re-predicting each step is the contract-clean equivalent. The trained chunk graph
  (a transformer emitting K actions) is exactly the artifact the edge-VLA distill+quantize step consumes.

torch lives here; the `core` contract stays numpy. The pure chunk-target construction is in `Demos.chunks`.
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
class ACTConfig:
    """ACT knobs — demo volume, transformer width/heads/depth, the chunk length, and the fit schedule.

    `chunk=8` is ~1/5 of the 40-step horizon: long enough to carry intent through the settle, short enough
    that the held-tail padding stays a small fraction of each window. `d_model=64` matches the BC MLP width so
    the paradigms are compared at equal capacity, not equal-plus."""
    n_demo_episodes: int = 200
    d_model: int = 64
    nhead: int = 4
    layers: int = 2
    chunk: int = 8
    epochs: int = 300
    lr: float = 1e-3
    max_steps: int = 40
    seed: int = 0


class ChunkTransformer(nn.Module):
    """Encode one observation, decode K action tokens — a minimal action-chunking transformer."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: ACTConfig):
        super().__init__()
        self._chunk = cfg.chunk
        self.obs_embed = nn.Linear(obs_dim, cfg.d_model)
        self.queries = nn.Parameter(torch.randn(cfg.chunk, cfg.d_model))
        layer = nn.TransformerDecoderLayer(cfg.d_model, cfg.nhead, dim_feedforward=cfg.d_model * 4,
                                           batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.layers)
        self.head = nn.Linear(cfg.d_model, act_dim)

    @override
    def forward(self, obs: Float[Tensor, "b obs"]) -> Float[Tensor, "b k act"]:
        memory = self.obs_embed(obs).unsqueeze(1)                    # (b, 1, d) — one observation token
        queries = self.queries.unsqueeze(0).expand(obs.shape[0], -1, -1)
        return self.head(self.decoder(queries, memory))


class ACTPolicy(Generic[StateT]):
    """Fit a chunk transformer on expert demos; execute the first action of each predicted chunk.

    Generic in the demonstrator's state type (`StateT`); its own state is the fitted `ChunkTransformer`."""

    def __init__(self, sim_env_factory: Callable[[int], Env], expert: ControlPolicy[StateT], cfg: ACTConfig):
        self._make_sim = sim_env_factory
        self._expert = expert
        self._cfg = cfg

    def train(self, task: str) -> ChunkTransformer:
        torch.manual_seed(self._cfg.seed)
        plan = EvalPlan(self._cfg.n_demo_episodes, self._cfg.seed, self._cfg.max_steps)
        obs, chunks = Demos.chunks(Demos.rollouts(self._expert, task, self._make_sim, plan), self._cfg.chunk)
        net = ChunkTransformer(obs.shape[1], chunks.shape[2], self._cfg)
        opt = torch.optim.Adam(net.parameters(), lr=self._cfg.lr)
        xb = torch.as_tensor(obs, dtype=torch.float32)
        yb = torch.as_tensor(chunks, dtype=torch.float32)
        for _ in range(self._cfg.epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb), yb)
            loss.backward()
            opt.step()
        return net

    def act(self, state: nn.Module, obs: Float[np.ndarray, "obs"]) -> Float[np.ndarray, "act"]:
        net = state
        with torch.no_grad():
            chunk = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return chunk[0, 0].numpy()                                   # first action of the predicted chunk
