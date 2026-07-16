"""Contract tests for core/policy.py — equivalence classes over the control boundary (no sim).

Classes: (1) protocol satisfaction — a minimal train/act object is a runtime instance of ControlPolicy;
(2) Trajectory names T-aligned per-timestep fields.
"""
import numpy as np

from core.policy import ControlPolicy, Trajectory


class _MiniPolicy:
    def train(self, task):
        return None

    def act(self, state, obs):
        return np.zeros(2)


class TestProtocolSatisfaction:
    def test_minimal_train_act_object_is_control_policy(self):
        assert isinstance(_MiniPolicy(), ControlPolicy)      # structural: has train + act

    def test_missing_act_is_not_a_control_policy(self):
        class _Half:
            def train(self, task):
                return None
        assert not isinstance(_Half(), ControlPolicy)


class TestTrajectoryAlignment:
    def test_all_fields_share_axis0_length_t(self):
        t = 5
        traj = Trajectory(
            obs=np.zeros((t, 4)),
            actions=np.zeros((t, 2)),
            rewards=np.zeros(t),
            dones=np.zeros(t, dtype=bool),
            success=np.zeros(t, dtype=bool),
        )
        assert {f.shape[0] for f in traj} == {t}
        assert traj._fields == ("obs", "actions", "rewards", "dones", "success")
