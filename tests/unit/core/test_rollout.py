"""Contract tests for core/rollout.py — the rollout spine driving a ControlPolicy through a stub env (no sim).

A deterministic `CountdownEnv` (counts n steps down to a terminal success) stands in for Isaac/MuJoCo, so
the spine's episode assembly, Trajectory alignment, and the sim-to-real policy-gap primitive are all
exercised with pure numpy. Classes: rollout assembles an axis-0-aligned Trajectory that ends on done;
success_rate / mean_return reduce trajectories; gap subtracts two rollout sets.
"""
import numpy as np

from core.policy import Trajectory
from core.reference_policy import ConstantActionPolicy
from core.rollout import Env, Rollout, StepResult


class CountdownEnv:
    """Reset to [n]; each step decrements; terminal (done+success) when it reaches 0."""

    def __init__(self, n: int):
        self._n0 = n
        self._n = n

    def reset(self) -> np.ndarray:
        self._n = self._n0
        return np.array([self._n], dtype=float)

    def step(self, action: np.ndarray) -> StepResult:
        self._n -= 1
        done = self._n <= 0
        return StepResult(obs=np.array([self._n], dtype=float), reward=1.0, done=done, success=done)


class TestEnvContract:
    def test_stub_env_satisfies_protocol(self):
        assert isinstance(CountdownEnv(3), Env)          # structural: has reset + step


class TestRollAssembly:
    def test_trajectory_is_axis0_aligned_and_ends_on_done(self):
        pol = ConstantActionPolicy(np.zeros(1))
        traj = Rollout.roll(pol, pol.train("countdown"), CountdownEnv(3), max_steps=10)
        assert isinstance(traj, Trajectory)
        assert {f.shape[0] for f in traj} == {3}         # 3 steps, stopped early on done (not max_steps)
        assert traj.dones[-1] and not traj.dones[:-1].any()
        assert traj.rewards.sum() == 3.0

    def test_max_steps_caps_a_nonterminating_episode(self):
        pol = ConstantActionPolicy(np.zeros(1))
        traj = Rollout.roll(pol, pol.train("countdown"), CountdownEnv(100), max_steps=5)
        assert traj.obs.shape[0] == 5 and not traj.dones.any()


class TestPolicyGap:
    def _trajs(self, success: bool, k: int) -> list[Trajectory]:
        return [Trajectory(np.zeros((2, 1)), np.zeros((2, 1)), np.ones(2), np.array([False, True]),
                           np.array([False, success])) for _ in range(k)]

    def test_success_rate_and_mean_return(self):
        trajs = self._trajs(success=True, k=4)
        assert Rollout.success_rate(trajs) == 1.0
        assert Rollout.mean_return(trajs) == 2.0
        assert Rollout.success_rate([]) == 0.0            # empty -> 0, no crash

    def test_gap_is_sim_minus_real(self):
        sim = self._trajs(success=True, k=2)              # sim solves it
        real = self._trajs(success=False, k=2)            # real does not
        assert Rollout.gap(sim, real) == 1.0              # fully sim-optimistic
