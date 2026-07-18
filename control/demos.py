"""Expert demonstrations through the rollout spine — the shared collection the imitation policies fit.

Behavior cloning wants flat `(obs, action)` pairs; the chunked paradigms (ACT, diffusion policy) want, for
each timestep, the observation and the *next K actions* — the action chunk they learn to predict. Both come
from the same expert rollouts, so one collector serves all three: `flat` concatenates per-step pairs,
`chunks` slides a K-window over each episode's actions (tail-padded by holding the last command, which for a
reach expert is the near-zero settle action). Pure numpy over `core.rollout` Trajectories — no torch here, so
`core` stays the numpy contract and the torch lives only inside each policy.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from jaxtyping import Float

from core.rollout import EvalPlan, Rollout, Trajectory


class Demos:
    """Collect an expert's rollouts once, expose them as flat BC pairs or as action-chunk targets."""

    @staticmethod
    def rollouts(expert: Any, task: str, make_env: Callable[[int], Any], plan: EvalPlan) -> list[Trajectory]:
        state = expert.train(task)
        return Rollout.rollset(expert, state, make_env, plan)

    @staticmethod
    def flat(trajs: list[Trajectory]) -> tuple[Float[np.ndarray, "n obs"], Float[np.ndarray, "n act"]]:
        """Per-step (obs, action) pairs — the behavior-cloning target."""
        return np.concatenate([t.obs for t in trajs]), np.concatenate([t.actions for t in trajs])

    @staticmethod
    def _chunk_one(actions: Float[np.ndarray, "t act"], horizon: int) -> Float[np.ndarray, "t k act"]:
        pad = np.repeat(actions[-1:], horizon, axis=0)          # hold last command past the episode end
        ext = np.concatenate([actions, pad], axis=0)
        return np.stack([ext[t:t + horizon] for t in range(len(actions))], axis=0)

    @staticmethod
    def chunks(trajs: list[Trajectory],
               horizon: int) -> tuple[Float[np.ndarray, "n obs"], Float[np.ndarray, "n k act"]]:
        """Per-step (obs, next-K-actions) — the action-chunk target ACT / diffusion policy predict."""
        obs = np.concatenate([t.obs for t in trajs])
        chunk = np.concatenate([Demos._chunk_one(t.actions, horizon) for t in trajs])
        return obs, chunk
