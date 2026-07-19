"""The DDPM noise schedule — the pure-numpy math a diffusion policy denoises against, kept out of the torch
model so it is unit-testable in isolation (Ho et al. 2020, the linear-β variance schedule).

A diffusion policy learns to reverse a fixed forward process that adds Gaussian noise to an action chunk over
`steps` levels. All of that process is closed-form in the per-step `betas`: the forward marginal
`x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε` and the reverse posterior mean both read their coefficients off the
cumulative product `ᾱ_t = Π α`. This class holds those coefficient arrays; the model file converts them to
tensors and only supplies the learned noise estimate `ε_θ`. Separating them means the schedule's invariants
(ᾱ monotone ↓, x_0-preservation at t=0, near-total noise at t=T) are checked without a GPU or a trained net.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float


class DiffusionSchedule:
    """A fixed linear-β DDPM schedule — the forward-noising and reverse-posterior coefficients as numpy."""

    def __init__(self, steps: int = 50, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.steps = steps
        self.betas = np.linspace(beta_start, beta_end, steps, dtype=np.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        # Precomputed forward coefficients for the TORCH trainer, which needs whole arrays to index with a
        # batched per-sample t (it cannot call q_sample: numpy, scalar t, no autograd). Deliberately NOT
        # shared with q_sample below — that one recomputes the formula, so the two are independent
        # implementations and `test_batched_torch_forward_matches_q_sample_reference` can actually catch a
        # drift here. Collapsing them onto one expression would make that test tautological.
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: Float[np.ndarray, "*shape"], t: int,
                 noise: Float[np.ndarray, "*shape"]) -> Float[np.ndarray, "*shape"]:
        """Forward marginal: noise a clean sample to level `t` in one closed-form step. The numpy
        REFERENCE for the process the torch trainer runs batched — the tensor path is the only one that
        runs in production, so this is the oracle a test cross-checks it against."""
        a_bar = self.alphas_cumprod[t]
        return np.sqrt(a_bar) * x0 + np.sqrt(1.0 - a_bar) * noise

    def posterior_mean(self, x_t: Float[np.ndarray, "*shape"], eps: Float[np.ndarray, "*shape"],
                       t: int) -> Float[np.ndarray, "*shape"]:
        """One reverse (denoising) step's mean given the model's noise estimate `eps` at level `t`."""
        a_t, a_bar = self.alphas[t], self.alphas_cumprod[t]
        return (x_t - self.betas[t] / np.sqrt(1.0 - a_bar) * eps) / np.sqrt(a_t)
