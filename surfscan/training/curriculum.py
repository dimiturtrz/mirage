"""Closed-loop curriculum — carried from the acoustic engine, ported to 3D defect synthesis.

The acoustic engine adapts generation difficulty per epoch from the trainer's previous-epoch score
(per-class SNR shift), coupling generator to trainer. Here the analog: adapt which **defect kind** the
synthesizer emits, biased toward the kinds the detector currently does *worst* on (highest training
loss). Weak kinds get sampled more → the generator chases the detector's weakness instead of a fixed
uniform mix. Almost nobody does this in 3D sim-to-real; it's the differentiator, not a reimplementation.

Design: single-kind batches so a batch's scalar loss attributes cleanly to one kind (an EMA per kind);
sample the next batch's kind ∝ softmax(standardized EMA loss). Unseen kinds start at the max seen loss
(explore-first). Set the temperature high to flatten toward uniform, low to chase the weakest hard.
"""
from __future__ import annotations

import numpy as np


class KindCurriculum:
    def __init__(self, kinds, seed: int = 0, alpha: float = 0.3, temp: float = 1.0):
        self.kinds = list(kinds)
        self.rng = np.random.RandomState(seed)
        self.alpha = alpha            # EMA weight on the newest loss
        self.temp = temp              # softmax temperature (low = chase weakest harder)
        self.loss: dict[str, float | None] = {k: None for k in self.kinds}
        self.history: list[dict] = []

    def _weights(self) -> np.ndarray:
        vals = [self.loss[k] for k in self.kinds]
        seen = [v for v in vals if v is not None]
        default = max(seen) if seen else 1.0                  # unseen kinds explore at max loss
        loss = np.array([v if v is not None else default for v in vals], dtype=float)
        z = (loss - loss.mean()) / (loss.std() + 1e-6) / self.temp
        w = np.exp(z)                                         # higher loss -> higher sampling prob
        return w / w.sum()

    def sample(self, n: int) -> tuple:
        """Return a single-kind tuple for the whole batch (clean per-kind loss attribution)."""
        return (str(self.rng.choice(self.kinds, p=self._weights())),)

    def observe(self, kinds: tuple, loss: float) -> None:
        k = kinds[0]
        prev = self.loss[k]
        self.loss[k] = loss if prev is None else (1 - self.alpha) * prev + self.alpha * loss

    def end_epoch(self) -> None:
        self.history.append({k: self.loss[k] for k in self.kinds})

    def weights(self) -> dict:
        """Current per-kind sampling probabilities (for logging/inspection)."""
        return dict(zip(self.kinds, self._weights().tolist(), strict=True))
