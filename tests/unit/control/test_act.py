"""Contract + smoke tests for control/act.py — the action-chunking transformer satisfies ControlPolicy and
learns to reach.

A tiny fit (few demos, few epochs, short chunk) keeps it fast. Equivalence classes: satisfies
core.policy.ControlPolicy; train->act round-trips a single shaped action (first of the chunk); the clone
reaches sim goals far better than a do-nothing policy.
"""
import numpy as np

from control.act import ACTConfig, ACTPolicy
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.policy import ControlPolicy
from core.rollout import Rollout


class TestACT:
    def _act(self) -> ACTPolicy:
        def make(seed: int) -> PointMassReach:
            return PointMassReach(Phys(), Task(), seed)
        return ACTPolicy(make, PDExpert(),
                         ACTConfig(n_demo_episodes=40, d_model=32, layers=1, chunk=4, epochs=120, seed=0))

    def test_is_control_policy(self):
        assert isinstance(self._act(), ControlPolicy)

    def test_train_then_act_returns_single_action(self):
        pol = self._act()
        state = pol.train("reach")
        action = pol.act(state, np.array([0.5, 0.5, 0.0, 0.0]))
        assert action.shape == (2,)

    def test_clone_reaches_sim(self):
        pol = self._act()
        state = pol.train("reach")
        trajs = [Rollout.roll(pol, state, PointMassReach(Phys(), Task(), 900 + i), max_steps=40) for i in range(20)]
        assert Rollout.success_rate(trajs) > 0.7
