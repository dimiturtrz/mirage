"""Contract + smoke tests for control/diffusion_policy.py — the conditional-DDPM policy satisfies ControlPolicy
and learns to reach.

A tiny fit (few demos/epochs, short chunk, few denoising steps) keeps it fast. Equivalence classes: satisfies
core.policy.ControlPolicy; train->act samples a chunk and executes one shaped action; the clone reaches sim
goals far better than a do-nothing policy.
"""
import numpy as np

from control.diffusion_policy import DiffusionConfig, DiffusionPolicy
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.policy import ControlPolicy
from core.rollout import Rollout


class TestDiffusionPolicy:
    def _diffusion(self) -> DiffusionPolicy:
        def make(seed: int) -> PointMassReach:
            return PointMassReach(Phys(), Task(), seed)
        return DiffusionPolicy(make, PDExpert(),
                               DiffusionConfig(n_demo_episodes=40, hidden=64, steps=20, chunk=4,
                                               epochs=200, seed=0))

    def test_is_control_policy(self):
        assert isinstance(self._diffusion(), ControlPolicy)

    def test_train_then_act_returns_single_action(self):
        pol = self._diffusion()
        state = pol.train("reach")
        action = pol.act(state, np.array([0.5, 0.5, 0.0, 0.0]))
        assert action.shape == (2,)

    def test_clone_reaches_sim(self):
        pol = self._diffusion()
        state = pol.train("reach")
        trajs = [Rollout.roll(pol, state, PointMassReach(Phys(), Task(), 900 + i), max_steps=40) for i in range(20)]
        assert Rollout.success_rate(trajs) > 0.6
