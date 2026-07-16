"""Contract tests for control/experiment.py — the mlflow-free compute paths produce the policy-gap metrics.

A tiny config keeps it fast. Equivalence classes: single-seed `_compute` carries the sim/expert successes
and a per-payload gap that grows with payload; multi-seed `_compute_multiseed` aggregates to mean/std keys.
"""
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

    def test_single_seed_metrics_carry_successes_and_per_payload_gap(self):
        m = Experiment._compute(self._cfg())
        assert m["expert_sim_success"] > 0.9              # analytic expert nearly perfect in-domain
        assert m["bc_sim_success"] > 0.7                  # clone learned the task
        for key in ("bc_sim_return", "bc_sim_steps",
                    "real_success_p110", "real_return_p110", "real_steps_p110", "gap_pp_p110",
                    "real_success_p160", "gap_pp_p160"):
            assert key in m

    def test_gap_grows_with_payload(self):
        m = Experiment._compute(self._cfg())
        assert m["gap_pp_p160"] >= m["gap_pp_p110"]       # heavier payload -> at least as large a gap

    def test_multiseed_aggregates_to_mean_and_std(self):
        agg = Experiment._compute_multiseed(self._cfg(n_seeds=2))
        for key in ("bc_sim_success_mean", "bc_sim_success_std", "gap_pp_p160_mean", "gap_pp_p160_std"):
            assert key in agg
        assert agg["gap_pp_p160_mean"] >= agg["gap_pp_p110_mean"]
        assert agg["gap_pp_p160_std"] >= 0.0
