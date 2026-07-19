"""Edge-VLA (hxq.4) — take the heaviest policy graph to the edge: distill the diffusion policy's 50-step
denoiser into a 1-step student, int8-quantize it, and measure what deployability costs in task success.

The differentiator the control leg claims is edge-VLA: a policy that runs real-time on-device, demonstrable
sim-in-the-loop without hardware. Two levers, both measured here on the point-mass reach:

- **Distillation** collapses the diffusion policy's iterative sampling (50 reverse steps per action) into a
  single feed-forward pass. `DistilledPolicy` is a 1-step student MLP that regresses the *teacher's sampled
  action chunk* — the standard diffusion→one-step distillation, at toy scale — so inference drops from 50
  network evaluations to 1 while the student still satisfies `core.policy.ControlPolicy`.
- **Quantization** takes the student to int8 (dynamic quantization of its linear layers), the weight form a
  fixed-function NPU actually runs, and we roll the quantized graph in sim/real to show the task survives.

The verdict axis is the deploy explorer's op-class (dense ships anywhere, attention doesn't fit a fixed NPU,
an iterative denoiser pays latency × steps). It is named here rather than imported from `surfscan.deploy` —
import-linter keeps control independent of surfscan; the unified cross-leg fit matrix is the deploy-substrate
generalization (j5s.6). Run: `python -m control.edge`. Footprints + quality logged to MLflow.
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, field
from typing import Any, ClassVar, override

import mlflow
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from control.demos import Demos
from control.diffusion_policy import DiffusionConfig, DiffusionPolicy
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.obs import Obs
from core.rollout import EvalPlan, Rollout

log = Obs.get()
_PP = 100.0
_KB = 1024


@dataclass(frozen=True)
class EdgeConfig:
    """Distillation + eval knobs. `distill_samples` caps the obs the student regresses the teacher on (each
    target costs one full 50-step teacher sample, so a subsample keeps distillation quick); `real` is the
    headline +50% payload / −10% gain shift the gap is read at."""
    teacher: DiffusionConfig = field(default_factory=DiffusionConfig)
    hidden: int = 128
    epochs: int = 400
    lr: float = 1e-3
    distill_samples: int = 3000
    episodes: int = 200
    eval_seed: int = 10_000
    sim: Phys = field(default_factory=Phys)
    real: Phys = field(default_factory=lambda: Phys(mass=1.5, gain=0.9))
    task: Task = field(default_factory=Task)
    seed: int = 0


class OneStepNet(nn.Module):
    """obs → flattened action chunk in a single forward — the distilled student's whole graph."""

    def __init__(self, obs_dim: int, chunk_flat: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, chunk_flat),
        )

    @override
    def forward(self, obs: Float[Tensor, "b obs"]) -> Float[Tensor, "b flat"]:
        return self.net(obs)


class DistilledPolicy:
    """A 1-step student regressing the diffusion teacher's sampled action chunk (50 denoise steps → 1 pass)."""

    def __init__(self, teacher: DiffusionPolicy, teacher_state: Any, expert: Any, make_sim: Any, cfg: EdgeConfig):
        self._teacher = teacher
        self._teacher_state = teacher_state
        self._expert = expert
        self._make_sim = make_sim
        self._cfg = cfg
        self._act_dim = 0

    def train(self, task: str) -> Any:
        torch.manual_seed(self._cfg.seed)
        rng = np.random.default_rng(self._cfg.seed)
        plan = EvalPlan(self._cfg.teacher.n_demo_episodes, self._cfg.seed, self._cfg.teacher.max_steps)
        obs = Demos.flat(Demos.rollouts(self._expert, task, self._make_sim, plan))[0]
        idx = rng.choice(len(obs), min(self._cfg.distill_samples, len(obs)), replace=False)
        obs_s = obs[idx]
        targets = np.stack([self._teacher.sample_chunk(self._teacher_state, o) for o in obs_s])   # (s, k, act)
        self._act_dim = targets.shape[2]
        flat = targets.reshape(len(obs_s), -1)
        net = OneStepNet(obs.shape[1], flat.shape[1], self._cfg.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self._cfg.lr)
        xb = torch.as_tensor(obs_s, dtype=torch.float32)
        yb = torch.as_tensor(flat, dtype=torch.float32)
        for _ in range(self._cfg.epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb), yb)
            loss.backward()
            opt.step()
        return net

    def act(self, state: Any, obs: Float[np.ndarray, "obs"]) -> Float[np.ndarray, "act"]:
        net = state
        with torch.no_grad():
            chunk = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return chunk.numpy().reshape(self._cfg.teacher.chunk, self._act_dim)[0]


class EdgeVLA:
    """Distill the diffusion teacher to a 1-step student, int8-quantize it, report footprint vs task success."""

    # op-class per paradigm — the deploy fit axis. Named here (not imported from surfscan.deploy): import-linter
    # keeps control independent of surfscan; the unified cross-leg fit matrix is the deploy generalization (j5s.6).
    _OP_CLASS: ClassVar[dict[str, str]] = {"diffusion_teacher": "iterative_dense",
                                           "distilled_fp32": "dense", "distilled_int8": "dense"}
    _STEPS: ClassVar[dict[str, int | None]] = {"diffusion_teacher": None, "distilled_fp32": 1, "distilled_int8": 1}

    @staticmethod
    def _serialized_kb(module: nn.Module) -> float:
        buf = io.BytesIO()
        torch.save(module.state_dict(), buf)
        return buf.tell() / _KB

    @staticmethod
    def _params_k(module: nn.Module) -> float:
        return sum(p.numel() for p in module.parameters()) / 1e3

    @staticmethod
    def quantize(net: nn.Module) -> nn.Module:
        """Dynamic int8 quantization of the student's linear layers — the weight form a fixed NPU runs."""
        return torch.ao.quantization.quantize_dynamic(net, {nn.Linear}, dtype=torch.qint8)

    @staticmethod
    def _quality(policy: Any, state: Any, cfg: EdgeConfig) -> tuple[float, float, float]:
        plan = EvalPlan(cfg.episodes, cfg.eval_seed, cfg.task.horizon)
        sim = Rollout.success_rate(Rollout.rollset(policy, state, PointMassReach.factory(cfg.sim, cfg.task), plan))
        real = Rollout.success_rate(Rollout.rollset(policy, state, PointMassReach.factory(cfg.real, cfg.task), plan))
        return sim, real, sim - real

    @staticmethod
    def _build(cfg: EdgeConfig) -> tuple[DiffusionPolicy, Any, DistilledPolicy, Any, Any]:
        make_sim = PointMassReach.factory(cfg.sim, cfg.task)
        expert = PDExpert(amax=cfg.sim.amax)
        teacher = DiffusionPolicy(make_sim, expert, cfg.teacher)
        teacher_state = teacher.train("reach")
        student = DistilledPolicy(teacher, teacher_state, expert, make_sim, cfg)
        student_state = student.train("reach")
        qnet = EdgeVLA.quantize(student_state)
        return teacher, teacher_state, student, student_state, qnet

    @staticmethod
    def _rows(cfg: EdgeConfig) -> dict[str, dict[str, float]]:
        teacher, teacher_state, student, student_state, qnet = EdgeVLA._build(cfg)
        # (policy, roll_state, param_net, size_net): int8 counts params off the fp32 net — quantization changes
        # precision, not count, and a quantized module packs weights into buffers so .parameters() reads empty.
        specs = {"diffusion_teacher": (teacher, teacher_state, teacher_state, teacher_state),
                 "distilled_fp32": (student, student_state, student_state, student_state),
                 "distilled_int8": (student, qnet, student_state, qnet)}
        rows: dict[str, dict[str, float]] = {}
        for name, (policy, roll_state, param_net, size_net) in specs.items():
            sim, real, gap = EdgeVLA._quality(policy, roll_state, cfg)
            rows[name] = {"params_k": EdgeVLA._params_k(param_net), "size_kb": EdgeVLA._serialized_kb(size_net),
                          "sim": sim, "real": real, "gap_pp": _PP * gap}
        return rows

    @staticmethod
    def run(cfg: EdgeConfig) -> dict[str, dict[str, float]]:
        rows = EdgeVLA._rows(cfg)
        EdgeVLA._report(cfg, rows)
        return rows

    @staticmethod
    def _report(cfg: EdgeConfig, rows: dict[str, dict[str, float]]) -> None:
        log.info("=== edge-VLA — distill (50-step diffusion → 1-step) + int8 quantize, +50%% shift ===")
        log.info("  policy              steps   params(k)  size(kB)   op-class         sim     real    gap(pp)")
        for name, row in rows.items():
            steps = EdgeVLA._STEPS[name]
            log.info("  %-18s  %5s   %8.1f   %7.1f   %-15s  %5.1f%%  %5.1f%%  %5.1f", name,
                     str(cfg.teacher.steps) if steps is None else str(steps), row["params_k"], row["size_kb"],
                     EdgeVLA._OP_CLASS[name], _PP * row["sim"], _PP * row["real"], row["gap_pp"])
        mlflow.set_experiment("control-policy-gap")
        with mlflow.start_run(run_name="edge-vla-distill-quantize"):
            mlflow.log_params({"distill_samples": cfg.distill_samples, "teacher_steps": cfg.teacher.steps,
                               "episodes": cfg.episodes})
            for name, row in rows.items():
                mlflow.log_metrics({f"{name}__{k}": v for k, v in row.items()})

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description="Edge-VLA: distill+quantize the diffusion policy to the edge.")
        ap.add_argument("--distill-samples", type=int, default=3000)
        ap.add_argument("--episodes", type=int, default=200)
        args = ap.parse_args()
        EdgeVLA.run(EdgeConfig(distill_samples=args.distill_samples, episodes=args.episodes))


if __name__ == "__main__":
    EdgeVLA.main()
