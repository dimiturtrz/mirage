"""Contract + smoke tests for control/diffusion_policy.py — the conditional-DDPM policy satisfies ControlPolicy
and learns to reach.

A tiny fit (few demos/epochs, short chunk, few denoising steps) keeps it fast. Equivalence classes: satisfies
core.policy.ControlPolicy; train->act samples a chunk and executes one shaped action; the clone reaches sim
goals far better than a do-nothing policy.
"""
import numpy as np
import torch

from control.diffusion_policy import DiffusionConfig, DiffusionPolicy
from control.diffusion_schedule import DiffusionSchedule
from control.expert import PDExpert
from control.point_mass import Phys, PointMassReach, Task
from core.policy import ControlPolicy
from core.rollout import Rollout


class TestDiffusionPolicy:
    def _diffusion(self) -> DiffusionPolicy:
        def make(seed: int) -> PointMassReach:
            return PointMassReach(Phys(), Task(), seed)
        return DiffusionPolicy(make, PDExpert(),
                               DiffusionConfig(n_demo_episodes=40, hidden=64, steps=20, chunk=4,
                                               epochs=200, seed=0))

    def test_is_control_policy(self):
        assert isinstance(self._diffusion(), ControlPolicy)

    def test_batched_torch_forward_matches_q_sample_reference(self):
        """The trainer noises a whole batch in torch with a PER-SAMPLE t; `DiffusionSchedule.q_sample` is
        the scalar-t numpy statement of that same forward marginal. Only the tensor path runs in
        production, so without this the reference and the real code can drift apart silently."""
        sched = DiffusionSchedule(steps=20)
        rng = np.random.default_rng(0)
        y0 = rng.standard_normal((5, 8)).astype(np.float32)
        noise = rng.standard_normal((5, 8)).astype(np.float32)
        levels = np.array([0, 3, 7, 12, 19])

        sqrt_ab = torch.as_tensor(sched.sqrt_alphas_cumprod, dtype=torch.float32)
        sqrt_1mab = torch.as_tensor(sched.sqrt_one_minus_alphas_cumprod, dtype=torch.float32)
        t = torch.as_tensor(levels)
        batched = (sqrt_ab[t].unsqueeze(1) * torch.as_tensor(y0)
                   + sqrt_1mab[t].unsqueeze(1) * torch.as_tensor(noise)).numpy()

        for i, level in enumerate(levels):
            assert np.allclose(batched[i], sched.q_sample(y0[i], int(level), noise[i]), atol=1e-5)

    def test_train_then_act_returns_single_action(self):
        pol = self._diffusion()
        state = pol.train("reach")
        action = pol.act(state, np.array([0.5, 0.5, 0.0, 0.0]))
        assert action.shape == (2,)

    def test_clone_reaches_sim(self):
        pol = self._diffusion()
        state = pol.train("reach")
        trajs = [Rollout.roll(pol, state, PointMassReach(Phys(), Task(), 900 + i), max_steps=40) for i in range(20)]
        assert Rollout.success_rate(trajs) > 0.6
