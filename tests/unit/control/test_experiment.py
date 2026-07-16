"""Contract test for control/experiment.py — the mlflow-free `_compute` produces the policy-gap metrics.

A tiny config keeps it fast. Equivalence classes: the metric dict carries the sim/expert successes and a
per-payload gap; the gap grows as the real payload grows (heavier payload -> larger sim-to-real gap).
"""
from control.experiment import Experiment, ExperimentConfig
from control.bc import BCConfig


class TestCompute:
    def _cfg(self) -> ExperimentConfig:
        return ExperimentConfig(
            bc=BCConfig(n_demo_episodes=40, hidden=32, epochs=150, seed=0),
            payload_sweep=(1.1, 1.6),
            eval_episodes=20,
        )

    def test_metrics_carry_successes_and_per_payload_gap(self):
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
