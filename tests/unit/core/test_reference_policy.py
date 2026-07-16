"""Contract tests for core/reference_policy.py — the trivial ControlPolicy proving the boundary runs (no sim).

Classes: the reference class is a runtime ControlPolicy; act emits the fixed action for any obs; the
closure form exposes the same behaviour.
"""
import numpy as np

from core.policy import ControlPolicy, Policy
from core.reference_policy import ConstantActionPolicy


class TestReferencePolicy:
    def test_is_control_policy(self):
        assert isinstance(ConstantActionPolicy(np.zeros(3)), ControlPolicy)

    def test_act_returns_fixed_action_for_any_obs(self):
        action = np.array([1.0, -1.0])
        pol = ConstantActionPolicy(action)
        state = pol.train("pick_place")
        assert state is None                            # stateless reference policy
        out = pol.act(state, np.random.default_rng(0).normal(size=17))
        assert np.array_equal(out, action)              # shaped, obs-independent

    def test_closure_form_is_control_policy(self):
        pol = ConstantActionPolicy(np.zeros(3)).closure()
        assert isinstance(pol, Policy)
        assert isinstance(pol, ControlPolicy)
