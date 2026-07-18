"""Compare the three imitation paradigms on one spine — behavior cloning, ACT, diffusion policy — cloned from
the *same* nominal-sim demos and measured through the *same* rollout / gap metric.

The point is not a leaderboard: on a unimodal reach the regressors already saturate, so this reports that all
three satisfy the `core.policy` contract and roll identically, and what their sim-to-real gap is at the
headline shift (+50% payload, −10% gain). It is a single-seed comparison by design — the multi-seed hardening
lives in `experiment.py` for the BC gap curve; a paradigm is only worth multi-seeding once a number moves
(house rule: quick-signal first, rigor once a number is worth hardening). Run: `python -m control.policy_compare`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import mlflow

from control.act import ACTConfig, ACTPolicy
from control.bc import BCConfig, BCPolicy
from control.diffusion_policy import DiffusionConfig, DiffusionPolicy
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.obs import Obs
from core.rollout import EvalPlan, Rollout

log = Obs.get()
_PP = 100.0


@dataclass(frozen=True)
class CompareConfig:
    """One shift, one seed, three policies — the cheap cross-paradigm check. `real` is the headline +50%
    payload / −10% actuator shift the BC curve identifies as the cliff; `episodes` matches `experiment.py`."""
    sim: Phys = field(default_factory=Phys)
    real: Phys = field(default_factory=lambda: Phys(mass=1.5, gain=0.9))
    task: Task = field(default_factory=Task)
    episodes: int = 200
    eval_seed: int = 10_000
    bc: BCConfig = field(default_factory=BCConfig)
    act: ACTConfig = field(default_factory=ACTConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


class PolicyCompare:
    """Train BC / ACT / diffusion on shared nominal demos, roll matched sim/real, report each one's gap."""

    @staticmethod
    def _policies(cfg: CompareConfig, train_make: Callable[[int], PointMassReach]) -> dict[str, Any]:
        expert = PDExpert(amax=cfg.sim.amax)
        return {"bc": BCPolicy(train_make, expert, cfg.bc),
                "act": ACTPolicy(train_make, expert, cfg.act),
                "diffusion": DiffusionPolicy(train_make, expert, cfg.diffusion)}

    @staticmethod
    def _gap(cfg: CompareConfig, policy: Any) -> dict[str, float]:
        state = policy.train("reach")
        plan = EvalPlan(cfg.episodes, cfg.eval_seed, cfg.task.horizon)
        sim = Rollout.success_rate(Rollout.rollset(policy, state, PointMassReach.factory(cfg.sim, cfg.task), plan))
        real = Rollout.success_rate(Rollout.rollset(policy, state, PointMassReach.factory(cfg.real, cfg.task), plan))
        return {"sim_success": sim, "real_success": real, "gap_pp": _PP * (sim - real)}

    @staticmethod
    def run(cfg: CompareConfig) -> dict[str, dict[str, float]]:
        train_make = PointMassReach.factory(cfg.sim, cfg.task)
        results = {name: PolicyCompare._gap(cfg, policy)
                   for name, policy in PolicyCompare._policies(cfg, train_make).items()}
        PolicyCompare._report(cfg, results)
        return results

    @staticmethod
    def _report(cfg: CompareConfig, results: dict[str, dict[str, float]]) -> None:
        mlflow.set_experiment("control-policy-gap")
        with mlflow.start_run(run_name="paradigm-compare-bc-act-diffusion"):
            mlflow.log_params({"episodes": cfg.episodes, "act_chunk": cfg.act.chunk,
                               "diffusion_steps": cfg.diffusion.steps})
            for name, row in results.items():
                mlflow.log_metrics({f"{name}__{k}": v for k, v in row.items()})
        log.info("=== paradigm compare — same nominal demos, +50%% payload / -10%% gain, single seed ===")
        log.info("  policy      sim success   real success   sim-to-real gap")
        for name, row in results.items():
            log.info("  %-10s   %6.1f%%       %6.1f%%        %5.1f pp", name,
                     _PP * row["sim_success"], _PP * row["real_success"], row["gap_pp"])

    @staticmethod
    def main() -> None:
        PolicyCompare.run(CompareConfig())


if __name__ == "__main__":
    PolicyCompare.main()
