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

    def q_sample(self, x0: Float[np.ndarray, "*shape"], t: int,
                 noise: Float[np.ndarray, "*shape"]) -> Float[np.ndarray, "*shape"]:
        """Forward marginal: noise a clean sample to level `t` in one closed-form step."""
        a_bar = self.alphas_cumprod[t]
        return np.sqrt(a_bar) * x0 + np.sqrt(1.0 - a_bar) * noise

    def posterior_mean(self, x_t: Float[np.ndarray, "*shape"], eps: Float[np.ndarray, "*shape"],
                       t: int) -> Float[np.ndarray, "*shape"]:
        """One reverse (denoising) step's mean given the model's noise estimate `eps` at level `t`."""
        a_t, a_bar = self.alphas[t], self.alphas_cumprod[t]
        return (x_t - self.betas[t] / np.sqrt(1.0 - a_bar) * eps) / np.sqrt(a_t)
