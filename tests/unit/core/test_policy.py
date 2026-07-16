"""Contract tests for core/policy.py — equivalence classes over the control boundary (no sim).

Classes: (1) protocol satisfaction — a `Policy` closure pair is a runtime instance of ControlPolicy;
(2) the closure round-trips train/act; (3) Trajectory names T-aligned per-timestep fields.
"""
import numpy as np

from core.policy import ControlPolicy, Policy, Trajectory


class TestProtocolSatisfaction:
    def test_closure_pair_is_control_policy(self):
        pol = Policy(train=lambda task: None, act=lambda state, obs: np.zeros(2))
        assert isinstance(pol, Policy)
        assert isinstance(pol, ControlPolicy)          # structural: has train + act


class TestClosureRoundtrips:
    def test_train_then_act(self):
        pol = Policy(train=lambda task: {"task": task}, act=lambda state, obs: np.full(2, state["task"] == "pp"))
        state = pol.train("pp")
        assert state == {"task": "pp"}
        out = pol.act(state, np.random.default_rng(0).normal(size=17))
        assert out.shape == (2,) and out.dtype == bool


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
