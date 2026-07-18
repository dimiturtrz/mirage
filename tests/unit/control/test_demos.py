"""Unit tests for control/demos.py — the shared expert-demo collector: flat BC pairs and padded action chunks.

Equivalence classes: flat concatenates per-step pairs to matching row counts; a single-episode chunk window
has shape (T, K, act) and its tail is padded by holding the last action; chunks over many episodes align row
counts with flat.
"""
import numpy as np

from control.demos import Demos
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.rollout import EvalPlan, Trajectory


def _trajs(n: int) -> list[Trajectory]:
    make = PointMassReach.factory(Phys(), Task())
    return Demos.rollouts(PDExpert(), "reach", make, EvalPlan(n, 0, 40))


class TestDemos:
    def test_flat_row_counts_match(self):
        trajs = _trajs(3)
        obs, act = Demos.flat(trajs)
        assert obs.shape[0] == act.shape[0] == sum(len(t.obs) for t in trajs)
        assert obs.shape[1] == 4 and act.shape[1] == 2

    def test_chunk_one_shape_and_tail_hold(self):
        actions = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])   # T=3, act=2
        chunk = Demos._chunk_one(actions, horizon=4)
        assert chunk.shape == (3, 4, 2)
        assert np.allclose(chunk[0], [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 1.0]])  # tail holds last
        assert np.allclose(chunk[-1], [[0.0, 1.0]] * 4)            # last step is all-held

    def test_chunks_align_with_flat(self):
        trajs = _trajs(3)
        obs_c, chunks = Demos.chunks(trajs, horizon=8)
        obs_f, _ = Demos.flat(trajs)
        assert obs_c.shape[0] == chunks.shape[0] == obs_f.shape[0]
        assert chunks.shape[1:] == (8, 2)
