"""hxq.8 — re-measure the sim-to-real policy gap at PhysX fidelity and compare to the numpy point mass.

Boots Isaac Sim, then runs the *same* control experiment as `control.experiment` but with the point-mass
env swapped for `IsaacReach` (a real PhysX rigid body). The policy (`control.bc.BCPolicy`), the expert
(`control.expert.PDExpert`), the rollout spine (`core.rollout.Rollout`), and the gap metric are imported
and used **unchanged** — only the `Env` implementation differs. That is the whole point of the `Env` seam:
the toy finding (a robustness margin, then a cliff) should transfer up the fidelity ladder without touching
policy/spine/metric code.

Reduced counts (fewer demos/episodes/seeds than the point-mass run) keep the GPU rollout minutes-long — this
is an opt-in fidelity check, not the CI-gated multi-seed harness. Run from sim/:

    OMNI_KIT_ACCEPT_EULA=YES uv run python isaac_gap.py [--seeds 3] [--episodes 40]
"""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from isaacsim import SimulationApp

_ARGS = argparse.ArgumentParser()
_ARGS.add_argument("--seeds", type=int, default=3)
_ARGS.add_argument("--episodes", type=int, default=40, help="eval + demo episodes per condition")
_ARGS.add_argument("--epochs", type=int, default=150, help="BC fit epochs")
ARGS = _ARGS.parse_args()

app = SimulationApp({"headless": True})
print("BOOTED", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> import core / control

import numpy as np  # noqa: E402

from control.bc import BCConfig, BCPolicy  # noqa: E402
from control.expert import PDExpert  # noqa: E402
from control.point_mass import Phys, Task  # noqa: E402  (reuse the physics/task param dataclasses)
from core.rollout import Rollout  # noqa: E402
from isaac_reach import IsaacReach  # noqa: E402

_PP = 100.0
PAYLOADS = (1.2, 1.4, 1.5, 1.6)
ACTUATOR_GAIN = 0.9
TASK = Task()
SIM = Phys()


class IsaacGap:
    """Drive the shared BC / expert / rollout stack over the reused IsaacReach env and report the gap curve."""

    def __init__(self, reach: IsaacReach):
        self._reach = reach

    def _factory(self, phys: Phys):
        def make(seed: int) -> IsaacReach:
            return self._reach.configure(seed, phys)
        return make

    def _success(self, policy, state, phys: Phys, episodes: int, seed0: int) -> float:
        make = self._factory(phys)
        trajs = [Rollout.roll(policy, state, make(seed0 + i), TASK.horizon) for i in range(episodes)]
        return Rollout.success_rate(trajs)

    def _one_seed(self, seed: int, episodes: int, epochs: int) -> dict[str, float]:
        expert = PDExpert(amax=SIM.amax)
        bc = BCPolicy(self._factory(SIM), expert,
                      BCConfig(n_demo_episodes=episodes, epochs=epochs, max_steps=TASK.horizon, seed=seed))
        state = bc.train("reach")
        sim_success = self._success(bc, state, SIM, episodes, seed0=10_000)
        row = {"bc_sim_success": sim_success}
        for payload in PAYLOADS:
            real = Phys(mass=payload, gain=ACTUATOR_GAIN)
            real_success = self._success(bc, state, real, episodes, seed0=10_000)
            row[f"real_success_p{round(payload * _PP)}"] = real_success
            row[f"gap_pp_p{round(payload * _PP)}"] = _PP * (sim_success - real_success)
        return row

    @staticmethod
    def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
        agg = {}
        for key in rows[0]:
            vals = np.array([r[key] for r in rows], dtype=float)
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return agg

    def run(self, seeds: int, episodes: int, epochs: int) -> dict[str, float]:
        rows = []
        for s in range(seeds):
            print(f"[seed {s}] training BC + rolling gap sweep...", flush=True)
            rows.append(self._one_seed(s, episodes, epochs))
        return self._aggregate(rows)

    @staticmethod
    def report(agg: dict[str, float], seeds: int) -> None:
        print(f"\n=== IsaacReach (PhysX) sim-to-real policy gap — {seeds} seeds, mean±sd ===", flush=True)
        print(f"BC sim success : {_PP * agg['bc_sim_success_mean']:5.1f} "
              f"± {_PP * agg['bc_sim_success_std']:.1f}%", flush=True)
        print("  payload     real success        sim-to-real gap (pp)", flush=True)
        for payload in PAYLOADS:
            tag = f"p{round(payload * _PP)}"
            print(f"  +{round((payload - 1.0) * _PP):3d}%      "
                  f"{_PP * agg[f'real_success_{tag}_mean']:5.1f} ± {_PP * agg[f'real_success_{tag}_std']:4.1f}%    "
                  f"    {agg[f'gap_pp_{tag}_mean']:5.1f} ± {agg[f'gap_pp_{tag}_std']:4.1f}", flush=True)
        print("\ncompare to point-mass (numpy Euler): +40% 17.8pp · +50% 60.0pp · +60% 98.0pp", flush=True)


def main() -> None:
    reach = IsaacReach.boot(TASK, phys_dt=SIM.dt)
    agg = IsaacGap(reach).run(ARGS.seeds, ARGS.episodes, ARGS.epochs)
    IsaacGap.report(agg, ARGS.seeds)
    app.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
