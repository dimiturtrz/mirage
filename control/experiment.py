"""The measured deliverable — train a behavior-cloned policy on nominal-sim demos, roll it in sim and in a
dynamics-shifted real, and report the **sim-to-real policy gap** (pp).

    policy gap = success_rate(sim) - success_rate(real)

Evaluation is matched: sim and real rollouts use the same goal seeds, so the gap isolates the dynamics
shift (heavier payload, weaker actuator) from goal luck. The expert's own sim success is logged as the
achievable ceiling. Run: `python -m control.experiment`. Metrics + params are logged to MLflow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import mlflow
import numpy as np

from control.bc import BCConfig, BCPolicy
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.obs import Obs
from core.rollout import Rollout, Trajectory

log = Obs.get()
_PP = 100.0  # fraction -> percentage points


@dataclass(frozen=True)
class EvalPlan:
    """How many matched episodes to evaluate, from which seed, over how long a horizon."""
    episodes: int
    seed0: int
    max_steps: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Nominal sim vs a swept real: a real robot arm carries an unmodeled payload (mass multiplier) under
    actuator sag (gain < 1). Sweeping the payload turns the gap into a graded curve, not one cherry-picked
    shift. Physically-argued defaults: +10 / +25 / +50% payload, -10% actuator authority."""
    sim: Phys = field(default_factory=Phys)
    task: Task = field(default_factory=Task)
    bc: BCConfig = field(default_factory=BCConfig)
    actuator_gain: float = 0.9
    payload_sweep: tuple[float, ...] = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
    eval_episodes: int = 200
    eval_seed: int = 10_000
    n_seeds: int = 5


class Experiment:
    """Train a sim-only BC policy and measure how far it falls as the real payload grows."""

    @staticmethod
    def _factory(phys: Phys, task: Task) -> Callable[[int], PointMassReach]:
        def make(seed: int) -> PointMassReach:
            return PointMassReach(phys, task, seed)
        return make

    @staticmethod
    def _rollset(policy: Any, state: Any, make_env: Callable[[int], Any], plan: EvalPlan) -> list[Trajectory]:
        return [Rollout.roll(policy, state, make_env(plan.seed0 + i), plan.max_steps)
                for i in range(plan.episodes)]

    @staticmethod
    def _steps_to_goal(trajs: list[Trajectory]) -> float:
        reached = [int(np.argmax(t.dones)) + 1 for t in trajs if t.dones.any()]   # first terminal step
        return float(np.mean(reached)) if reached else math.nan

    @staticmethod
    def _summary(trajs: list[Trajectory]) -> tuple[float, float, float]:
        return Rollout.success_rate(trajs), Rollout.mean_return(trajs), Experiment._steps_to_goal(trajs)

    @staticmethod
    def _real_summary(cfg: ExperimentConfig, bc: Any, state: Any, plan: EvalPlan,
                      payload: float) -> tuple[float, float, float]:
        real_phys = Phys(mass=payload, gain=cfg.actuator_gain)
        return Experiment._summary(Experiment._rollset(bc, state, Experiment._factory(real_phys, cfg.task), plan))

    @staticmethod
    def _compute(cfg: ExperimentConfig) -> dict[str, float]:
        sim_make = Experiment._factory(cfg.sim, cfg.task)
        plan = EvalPlan(cfg.eval_episodes, cfg.eval_seed, cfg.task.horizon)

        expert = PDExpert(amax=cfg.sim.amax)
        bc = BCPolicy(sim_make, expert, cfg.bc)
        state = bc.train("reach")

        sim_success, sim_return, sim_steps = Experiment._summary(Experiment._rollset(bc, state, sim_make, plan))
        expert_sim = Rollout.success_rate(Experiment._rollset(expert, expert.train("reach"), sim_make, plan))

        metrics = {"bc_sim_success": sim_success, "bc_sim_return": sim_return, "bc_sim_steps": sim_steps,
                   "expert_sim_success": expert_sim}
        for payload in cfg.payload_sweep:
            real_success, real_return, real_steps = Experiment._real_summary(cfg, bc, state, plan, payload)
            tag = f"p{round(payload * _PP)}"
            metrics[f"real_success_{tag}"] = real_success
            metrics[f"real_return_{tag}"] = real_return
            metrics[f"real_steps_{tag}"] = real_steps
            metrics[f"gap_pp_{tag}"] = _PP * (sim_success - real_success)
        return metrics

    @staticmethod
    def _aggregate(per_seed: list[dict[str, float]]) -> dict[str, float]:
        agg: dict[str, float] = {}
        for key in per_seed[0]:
            vals = np.array([ps[key] for ps in per_seed], dtype=float)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        return agg

    @staticmethod
    def _compute_multiseed(cfg: ExperimentConfig) -> dict[str, float]:
        per_seed = [Experiment._compute(replace(cfg, bc=replace(cfg.bc, seed=s))) for s in range(cfg.n_seeds)]
        return Experiment._aggregate(per_seed)

    @staticmethod
    def run(cfg: ExperimentConfig) -> dict[str, float]:
        agg = Experiment._compute_multiseed(cfg)
        Experiment._report(cfg, agg)
        return agg

    @staticmethod
    def _report(cfg: ExperimentConfig, agg: dict[str, float]) -> None:
        mlflow.set_experiment("control-policy-gap")
        with mlflow.start_run(run_name="point-mass-reach-bc"):
            mlflow.log_params({
                "actuator_gain": cfg.actuator_gain, "payload_sweep": str(cfg.payload_sweep),
                "success_eps": cfg.task.eps, "horizon": cfg.task.horizon,
                "bc_demos": cfg.bc.n_demo_episodes, "bc_epochs": cfg.bc.epochs,
                "eval_episodes": cfg.eval_episodes, "n_seeds": cfg.n_seeds,
            })
            mlflow.log_metrics({k: v for k, v in agg.items() if math.isfinite(v)})
        log.info("=== point-mass reach — sim-to-real policy gap (BC, sim-only; %d seeds, mean±sd) ===", cfg.n_seeds)
        log.info("expert sim success : %5.1f%%   (achievable ceiling)", _PP * agg["expert_sim_success_mean"])
        log.info("BC     sim success : %5.1f ± %.1f%%",
                 _PP * agg["bc_sim_success_mean"], _PP * agg["bc_sim_success_std"])
        log.info("  payload   real-success (mean±sd)    sim-to-real gap (pp)")
        for payload in cfg.payload_sweep:
            tag = f"p{round(payload * _PP)}"
            log.info("  +%3d%%      %5.1f ± %4.1f%%           %5.1f ± %4.1f",
                     round((payload - 1.0) * _PP),
                     _PP * agg[f"real_success_{tag}_mean"], _PP * agg[f"real_success_{tag}_std"],
                     agg[f"gap_pp_{tag}_mean"], agg[f"gap_pp_{tag}_std"])

    @staticmethod
    def main() -> None:
        Experiment.run(ExperimentConfig())


if __name__ == "__main__":
    Experiment.main()
