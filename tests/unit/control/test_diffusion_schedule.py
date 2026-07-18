"""Unit tests for control/diffusion_schedule.py — the pure DDPM coefficient math a diffusion policy denoises
against.

Equivalence classes: ᾱ is monotone decreasing over the chain (noise accumulates); the forward marginal at
t=0 barely perturbs a clean sample and preserves it exactly with zero noise; the reverse posterior mean is
shape-preserving and reduces to the plain re-scaling when the noise estimate is zero.
"""
import numpy as np

from control.diffusion_schedule import DiffusionSchedule


class TestDiffusionSchedule:
    def test_alphas_cumprod_monotone_decreasing(self):
        sched = DiffusionSchedule(steps=50)
        assert np.all(np.diff(sched.alphas_cumprod) < 0)

    def test_q_sample_zero_noise_is_scaled_x0(self):
        sched = DiffusionSchedule(steps=50)
        x0 = np.array([1.0, -2.0, 0.5])
        out = sched.q_sample(x0, t=10, noise=np.zeros_like(x0))
        assert np.allclose(out, np.sqrt(sched.alphas_cumprod[10]) * x0)

    def test_q_sample_t0_barely_perturbs(self):
        sched = DiffusionSchedule(steps=50)
        x0 = np.array([1.0, -2.0, 0.5])
        out = sched.q_sample(x0, t=0, noise=np.ones_like(x0))
        assert np.allclose(out, x0, atol=2e-2)                     # a_bar[0] ~ 1 -> almost no noise

    def test_posterior_mean_zero_eps_is_rescale(self):
        sched = DiffusionSchedule(steps=50)
        x_t = np.array([0.3, -0.7, 1.1])
        out = sched.posterior_mean(x_t, eps=np.zeros_like(x_t), t=20)
        assert out.shape == x_t.shape
        assert np.allclose(out, x_t / np.sqrt(sched.alphas[20]))
