"""Contract + physics tests for control/point_mass.py — the reach env under the rollout spine's Env contract.

Equivalence classes: env satisfies core.rollout.Env; reset frames the goal relative to a rest start;
one Newtonian step integrates accel->vel->pos; the episode terminates on success or at the horizon.
"""
import numpy as np
import pytest

from control.point_mass import Phys, PointMassReach, Task
from core.rollout import Env


class TestEnvContract:
    def test_satisfies_env_protocol(self):
        assert isinstance(PointMassReach(Phys(), Task(), seed=0), Env)


class TestReset:
    def test_obs_is_goal_relative_from_rest(self):
        env = PointMassReach(Phys(), Task(goal_radius=1.0), seed=0)
        obs = env.reset()
        assert obs.shape == (4,)
        assert np.linalg.norm(obs[:2]) == pytest.approx(1.0)   # pos=0 -> obs[:2] = goal on the unit circle
        assert np.array_equal(obs[2:], np.zeros(2))            # vel starts at rest

    def test_same_seed_same_goal(self):
        a = PointMassReach(Phys(), Task(), seed=7).reset()
        b = PointMassReach(Phys(), Task(), seed=7).reset()
        assert np.array_equal(a, b)


class TestStep:
    def test_one_step_integrates_newtonian_dynamics(self):
        env = PointMassReach(Phys(mass=1.0, gain=1.0, drag=0.1, dt=0.05, amax=1.0), Task(), seed=0)
        env.reset()
        step = env.step(np.array([1.0, 0.0]))
        # acc = (1*1 - 0.1*0)/1 = [1,0]; vel = acc*dt = [0.05,0]; pos = vel*dt = [0.0025,0]
        assert step.obs[2:] == pytest.approx([0.05, 0.0])       # obs[2:] is velocity
        assert not step.done

    def test_horizon_terminates_a_never_reaching_episode(self):
        env = PointMassReach(Phys(), Task(horizon=3, eps=0.001), seed=0)
        env.reset()
        results = [env.step(np.zeros(2)) for _ in range(3)]      # zero action from rest never reaches
        assert not results[0].done and not results[1].done
        assert results[2].done and not results[2].success        # done by horizon, not by success
