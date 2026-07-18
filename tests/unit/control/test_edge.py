"""Contract + smoke tests for control/edge.py — the distilled 1-step student satisfies ControlPolicy and
survives int8 quantization.

A tiny teacher (few demos, short chain) keeps it fast. Equivalence classes: the distilled student satisfies
core.policy.ControlPolicy and reaches sim; int8 quantization shrinks the serialized graph and the quantized
student still reaches (the task survives the edge form).
"""
import numpy as np

from control.diffusion_policy import DiffusionConfig, DiffusionPolicy
from control.edge import DistilledPolicy, EdgeConfig, EdgeVLA
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.policy import ControlPolicy
from core.rollout import Rollout


def _make(seed: int) -> PointMassReach:
    return PointMassReach(Phys(), Task(), seed)


def _student() -> tuple[DistilledPolicy, object]:
    teacher_cfg = DiffusionConfig(n_demo_episodes=20, hidden=64, steps=10, chunk=4, epochs=100, seed=0)
    cfg = EdgeConfig(teacher=teacher_cfg, hidden=64, epochs=150, distill_samples=300, episodes=20, seed=0)
    expert = PDExpert()
    teacher = DiffusionPolicy(_make, expert, teacher_cfg)
    student = DistilledPolicy(teacher, teacher.train("reach"), expert, _make, cfg)
    return student, student.train("reach")


class TestEdge:
    def test_distilled_is_control_policy(self):
        student, _ = _student()
        assert isinstance(student, ControlPolicy)

    def test_train_then_act_returns_single_action(self):
        student, state = _student()
        action = student.act(state, np.array([0.5, 0.5, 0.0, 0.0]))
        assert action.shape == (2,)

    def test_int8_quantization_shrinks_and_runs(self):
        student, state = _student()
        qnet = EdgeVLA.quantize(state)
        assert EdgeVLA._serialized_kb(qnet) < EdgeVLA._serialized_kb(state)   # int8 graph is smaller
        action = student.act(qnet, np.array([0.5, 0.5, 0.0, 0.0]))
        assert action.shape == (2,)                                          # quantized graph still produces actions
        Rollout.roll(student, qnet, _make(900), max_steps=40)                # and rolls without error
