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

from control.adaptive import AdaptiveExpert, ProprioEnv
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
    dr_mass_range: tuple[float, float] = (1.0, 1.6)
    dr_gain_range: tuple[float, float] = (0.85, 1.0)
    eval_episodes: int = 200
    eval_seed: int = 10_000
    n_seeds: int = 5


@dataclass(frozen=True)
class Arm:
    """One training regime to measure the gap under: a demonstrator, an env wrapper (identity or the
    proprioceptive augmentation), and the demo-collection dynamics. The three arms — nominal (PD, blind,
    nominal demos), dr (PD, blind, randomized demos), adaptive (online system-ID, proprioceptive obs,
    randomized demos) — differ only in these three fields; everything downstream is shared."""
    name: str
    expert: Any
    wrap: Callable[[Any], Any]
    train_make: Callable[[int], PointMassReach]


@dataclass(frozen=True)
class Trained:
    """A trained policy ready to roll — the fit `policy`, its `state`, and the env `wrap` its observation
    space was built for (identity or proprioceptive). Travels as one handle so eval helpers stay small."""
    policy: Any
    state: Any
    wrap: Callable[[Any], Any]


class Experiment:
    """Train a sim-only BC policy and measure how far it falls as the real payload grows."""

    @staticmethod
    def _factory(phys: Phys, task: Task) -> Callable[[int], PointMassReach]:
        def make(seed: int) -> PointMassReach:
            return PointMassReach(phys, task, seed)
        return make

    @staticmethod
    def _wrap_factory(make: Callable[[int], Any], wrap: Callable[[Any], Any]) -> Callable[[int], Any]:
        def wrapped(seed: int) -> Any:
            return wrap(make(seed))
        return wrapped

    @staticmethod
    def _dr_factory(cfg: ExperimentConfig) -> Callable[[int], PointMassReach]:
        """Domain-randomized demo env: each episode samples payload mass + actuator gain from the training
        ranges, so the cloned policy sees a spread of dynamics instead of nominal-only. The goal still rides
        `seed` (same seed -> same goal as nominal), isolating the dynamics randomization from goal luck."""
        lo_m, hi_m = cfg.dr_mass_range
        lo_g, hi_g = cfg.dr_gain_range

        def make(seed: int) -> PointMassReach:
            rng = np.random.default_rng(seed)
            phys = Phys(mass=float(rng.uniform(lo_m, hi_m)), gain=float(rng.uniform(lo_g, hi_g)))
            return PointMassReach(phys, cfg.task, seed)
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
    def _real_summary(cfg: ExperimentConfig, trained: Trained, plan: EvalPlan,
                      payload: float) -> tuple[float, float, float]:
        real_phys = Phys(mass=payload, gain=cfg.actuator_gain)
        make = Experiment._wrap_factory(Experiment._factory(real_phys, cfg.task), trained.wrap)
        return Experiment._summary(Experiment._rollset(trained.policy, trained.state, make, plan))

    @staticmethod
    def _compute(cfg: ExperimentConfig, arm: Arm) -> dict[str, float]:
        sim_make = Experiment._wrap_factory(Experiment._factory(cfg.sim, cfg.task), arm.wrap)  # nominal eval sim
        train_make = Experiment._wrap_factory(arm.train_make, arm.wrap)
        plan = EvalPlan(cfg.eval_episodes, cfg.eval_seed, cfg.task.horizon)

        bc = BCPolicy(train_make, arm.expert, cfg.bc)
        trained = Trained(bc, bc.train("reach"), arm.wrap)

        sim_success, sim_return, sim_steps = Experiment._summary(
            Experiment._rollset(trained.policy, trained.state, sim_make, plan))
        expert_sim = Rollout.success_rate(
            Experiment._rollset(arm.expert, arm.expert.train("reach"), sim_make, plan))

        metrics = {"bc_sim_success": sim_success, "bc_sim_return": sim_return, "bc_sim_steps": sim_steps,
                   "expert_sim_success": expert_sim}
        for payload in cfg.payload_sweep:
            real_success, real_return, real_steps = Experiment._real_summary(cfg, trained, plan, payload)
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
    def _compute_multiseed(cfg: ExperimentConfig, make_arm: Callable[[ExperimentConfig], Arm]) -> dict[str, float]:
        seeded = [replace(cfg, bc=replace(cfg.bc, seed=s)) for s in range(cfg.n_seeds)]
        return Experiment._aggregate([Experiment._compute(c, make_arm(c)) for c in seeded])

    @staticmethod
    def _identity(env: Any) -> Any:
        return env

    @staticmethod
    def _nominal_arm(cfg: ExperimentConfig) -> Arm:
        return Arm("nominal", PDExpert(amax=cfg.sim.amax), Experiment._identity,
                   Experiment._factory(cfg.sim, cfg.task))

    @staticmethod
    def _dr_arm(cfg: ExperimentConfig) -> Arm:
        return Arm("dr", PDExpert(amax=cfg.sim.amax), Experiment._identity, Experiment._dr_factory(cfg))

    @staticmethod
    def _adaptive_arm(cfg: ExperimentConfig) -> Arm:
        return Arm("adaptive", AdaptiveExpert(dt=cfg.sim.dt, amax=cfg.sim.amax), ProprioEnv,
                   Experiment._dr_factory(cfg))

    @staticmethod
    def run(cfg: ExperimentConfig) -> dict[str, dict[str, float]]:
        builders = {"nominal": Experiment._nominal_arm, "dr": Experiment._dr_arm,
                    "adaptive": Experiment._adaptive_arm}
        arms = {name: Experiment._compute_multiseed(cfg, build) for name, build in builders.items()}
        Experiment._report(cfg, arms)
        return arms

    @staticmethod
    def _report(cfg: ExperimentConfig, arms: dict[str, dict[str, float]]) -> None:
        mlflow.set_experiment("control-policy-gap")
        with mlflow.start_run(run_name="point-mass-reach-bc-arms"):
            mlflow.log_params({
                "actuator_gain": cfg.actuator_gain, "payload_sweep": str(cfg.payload_sweep),
                "dr_mass_range": str(cfg.dr_mass_range), "dr_gain_range": str(cfg.dr_gain_range),
                "success_eps": cfg.task.eps, "horizon": cfg.task.horizon,
                "bc_demos": cfg.bc.n_demo_episodes, "bc_epochs": cfg.bc.epochs,
                "eval_episodes": cfg.eval_episodes, "n_seeds": cfg.n_seeds,
            })
            for arm, agg in arms.items():
                mlflow.log_metrics({f"{arm}__{k}": v for k, v in agg.items() if math.isfinite(v)})
        Experiment._log_board(cfg, arms)

    @staticmethod
    def _log_board(cfg: ExperimentConfig, arms: dict[str, dict[str, float]]) -> None:
        nom, dr, ad = arms["nominal"], arms["dr"], arms["adaptive"]
        log.info("=== point-mass reach — sim-to-real policy gap: DR + adaptive levers "
                 "(BC, sim-only; %d seeds, mean±sd) ===", cfg.n_seeds)
        log.info("expert sim success  : %5.1f%%   (achievable ceiling)", _PP * nom["expert_sim_success_mean"])
        log.info("BC sim success      : nominal %5.1f%%   dr %5.1f%%   adaptive %5.1f%%",
                 _PP * nom["bc_sim_success_mean"], _PP * dr["bc_sim_success_mean"], _PP * ad["bc_sim_success_mean"])
        log.info("  payload    nominal gap        dr gap            adaptive gap (pp)")
        for payload in cfg.payload_sweep:
            tag = f"gap_pp_p{round(payload * _PP)}"
            log.info("  +%3d%%     %5.1f ± %4.1f      %5.1f ± %4.1f      %5.1f ± %4.1f",
                     round((payload - 1.0) * _PP),
                     nom[f"{tag}_mean"], nom[f"{tag}_std"], dr[f"{tag}_mean"], dr[f"{tag}_std"],
                     ad[f"{tag}_mean"], ad[f"{tag}_std"])

    @staticmethod
    def main() -> None:
        Experiment.run(ExperimentConfig())


if __name__ == "__main__":
    Experiment.main()
