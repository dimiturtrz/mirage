"""Unit tests for control/adaptive.py — the proprioceptive wrapper + the online system-ID controller.

Equivalence classes: ProprioEnv widens the obs to 8-dim and carries [last_action, Δv] across a step;
AdaptiveExpert reduces to nominal PD when there is no history, and commands *harder* under an estimated
heavy plant (k̂ < 1) than the un-compensated PD would.
"""
import numpy as np

from control.adaptive import AdaptiveExpert, ProprioEnv
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task


class TestProprioEnv:
    def _env(self, mass: float = 1.0) -> ProprioEnv:
        return ProprioEnv(PointMassReach(Phys(mass=mass), Task(), seed=0))

    def test_reset_widens_obs_to_8_with_zero_history(self):
        obs = self._env().reset()
        assert obs.shape == (8,)
        assert (obs[4:] == 0.0).all()                    # [last_action, Δv] start at zero

    def test_step_carries_last_action_and_velocity_change(self):
        env = self._env()
        env.reset()
        action = np.array([0.7, -0.3])
        result = env.step(action)
        assert result.obs.shape == (8,)
        assert np.allclose(result.obs[4:6], action)      # last_action echoed
        assert np.allclose(result.obs[6:8], result.obs[2:4])  # Δv == vel (started from rest)


class TestAdaptiveExpert:
    def _expert(self) -> AdaptiveExpert:
        return AdaptiveExpert(dt=Phys().dt)

    def test_no_history_reduces_to_nominal_pd(self):
        obs8 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # goal ahead, at rest, no history
        pd = PDExpert()
        adaptive = self._expert().act(None, obs8)
        assert np.allclose(adaptive, pd.act(None, obs8[:4]))         # k̂=1 -> identical command

    def test_estimated_heavy_plant_raises_the_command(self):
        # a close goal keeps the nominal command unsaturated so the gain-inversion is visible; the same
        # action producing a smaller Δv reads as a heavier plant (k̂ < 1) and lifts the command.
        base = np.array([0.3, 0.0, 0.0, 0.0])
        heavy = np.concatenate([base, [1.0, 0.0], [0.03, 0.0]])      # big action, small Δv => k̂ = 0.6
        light = np.concatenate([base, [1.0, 0.0], [0.05, 0.0]])      # same action, larger Δv => k̂ = 1.0
        expert = self._expert()
        assert abs(expert.act(None, heavy)[0]) > abs(expert.act(None, light)[0])
