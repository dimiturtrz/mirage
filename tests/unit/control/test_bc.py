"""Contract + smoke tests for control/bc.py — behavior cloning satisfies ControlPolicy and learns to reach.

A tiny fit (few demos, few epochs) keeps it fast. Equivalence classes: satisfies core.policy.ControlPolicy;
train->act round-trips a shaped action; the sim-trained clone reaches sim goals far better than a do-nothing
policy (the clone actually learned the task, not just the interface).
"""
import numpy as np

from control.bc import BCConfig, BCPolicy
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.policy import ControlPolicy
from core.rollout import Rollout


class TestBC:
    def _bc(self) -> BCPolicy:
        def make(seed: int) -> PointMassReach:
            return PointMassReach(Phys(), Task(), seed)
        return BCPolicy(make, PDExpert(), BCConfig(n_demo_episodes=40, hidden=32, epochs=150, seed=0))

    def test_is_control_policy(self):
        assert isinstance(self._bc(), ControlPolicy)

    def test_train_then_act_returns_shaped_action(self):
        bc = self._bc()
        state = bc.train("reach")
        action = bc.act(state, np.array([0.5, 0.5, 0.0, 0.0]))
        assert action.shape == (2,)

    def test_clone_reaches_sim_far_better_than_zero_policy(self):
        bc = self._bc()
        state = bc.train("reach")
        trajs = [Rollout.roll(bc, state, PointMassReach(Phys(), Task(), 900 + i), max_steps=40) for i in range(20)]
        assert Rollout.success_rate(trajs) > 0.7           # a do-nothing policy scores ~0 here
