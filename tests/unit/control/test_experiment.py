"""Contract tests for control/experiment.py — the mlflow-free compute paths produce the policy-gap metrics.

A tiny config keeps it fast. Equivalence classes: single-seed `_compute` carries the sim/expert successes
and a per-payload gap that grows with payload; multi-seed `_compute_multiseed` aggregates to mean/std keys.
"""
import numpy as np

from control.bc import BCConfig
from control.experiment import Experiment, ExperimentConfig


class TestCompute:
    def _cfg(self, n_seeds: int = 1) -> ExperimentConfig:
        return ExperimentConfig(
            bc=BCConfig(n_demo_episodes=40, hidden=32, epochs=150, seed=0),
            payload_sweep=(1.1, 1.6),
            eval_episodes=20,
            n_seeds=n_seeds,
        )

    def _nominal(self, cfg: ExperimentConfig):
        return Experiment._compute(cfg, Experiment._nominal_arm(cfg))

    def test_single_seed_metrics_carry_successes_and_per_payload_gap(self):
        m = self._nominal(self._cfg())
        assert m["expert_sim_success"] > 0.9              # analytic expert nearly perfect in-domain
        assert m["bc_sim_success"] > 0.7                  # clone learned the task
        for key in ("bc_sim_return", "bc_sim_steps",
                    "real_success_p110", "real_return_p110", "real_steps_p110", "gap_pp_p110",
                    "real_success_p160", "gap_pp_p160"):
            assert key in m

    def test_gap_grows_with_payload(self):
        m = self._nominal(self._cfg())
        assert m["gap_pp_p160"] >= m["gap_pp_p110"]       # heavier payload -> at least as large a gap

    def _step_once(self, env):
        env.reset()
        return env.step(np.array([1.0, 1.0])).obs                   # obs after one fixed action from reset

    def test_dr_factory_randomizes_dynamics_but_keeps_goal(self):
        cfg = self._cfg()
        make = Experiment._dr_factory(cfg)
        nominal = Experiment._factory(cfg.sim, cfg.task)
        assert (make(0).reset()[:2] == nominal(0).reset()[:2]).all()          # goal still rides seed
        dr_obs, dr_again = self._step_once(make(0)), self._step_once(make(0))
        assert (dr_obs == dr_again).all()                                     # seed-deterministic dynamics
        assert not (dr_obs[2:] == self._step_once(nominal(0))[2:]).all()      # different Phys -> different vel

    def test_multiseed_aggregates_to_mean_and_std(self):
        agg = Experiment._compute_multiseed(self._cfg(n_seeds=2), Experiment._nominal_arm)
        for key in ("bc_sim_success_mean", "bc_sim_success_std", "gap_pp_p160_mean", "gap_pp_p160_std"):
            assert key in agg
        assert agg["gap_pp_p160_mean"] >= agg["gap_pp_p110_mean"]
        assert agg["gap_pp_p160_std"] >= 0.0

    def test_dr_arm_computes_the_full_gap_curve(self):
        agg = Experiment._compute_multiseed(self._cfg(n_seeds=1), Experiment._dr_arm)
        for key in ("bc_sim_success_mean", "gap_pp_p110_mean", "gap_pp_p160_mean"):
            assert key in agg

    def test_adaptive_arm_uses_proprio_obs_and_computes_the_gap_curve(self):
        agg = Experiment._compute_multiseed(self._cfg(n_seeds=1), Experiment._adaptive_arm)
        for key in ("bc_sim_success_mean", "gap_pp_p110_mean", "gap_pp_p160_mean"):
            assert key in agg
        assert agg["bc_sim_success_mean"] > 0.7          # the adaptive clone still solves nominal sim
