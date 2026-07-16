"""Contract tests for control/expert.py — the analytic PD demonstrator satisfies ControlPolicy and steers.

Equivalence classes: satisfies core.policy.ControlPolicy; act drives toward the goal; the action saturates
at the clip bound for a large error.
"""
import numpy as np

from control.expert import PDExpert
from core.policy import ControlPolicy


class TestExpert:
    def test_is_control_policy(self):
        assert isinstance(PDExpert(), ControlPolicy)

    def test_act_steers_toward_goal(self):
        pol = PDExpert(kp=1.5, kd=1.2, amax=10.0)          # high clip so the raw PD term shows through
        obs = np.array([0.4, -0.2, 0.0, 0.0])               # goal-pos = [0.4,-0.2], vel = 0
        action = pol.act(None, obs)
        assert action.shape == (2,)
        assert action[0] > 0 and action[1] < 0             # push toward the goal offset

    def test_action_saturates_at_clip(self):
        pol = PDExpert(kp=1.5, kd=1.2, amax=1.0)
        action = pol.act(None, np.array([100.0, 0.0, 0.0, 0.0]))   # huge error
        assert np.allclose(action, [1.0, 0.0])             # clipped to amax
