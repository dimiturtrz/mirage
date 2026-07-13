"""One SGD loop — shuffle, minibatch, zero/backward/step — shared by every trained method.

The method supplies model + optimizer + a `step_fn(idx) -> loss` tensor that owns batch assembly, the
forward, any autocast, and the loss; the Trainer owns only the epoch/minibatch/optimizer mechanics that
were hand-rolled four times (draem, featrecon, triad, the vae in train.py). Device-injected, no
torchvision — so it's unit-testable on a tiny cpu model.

    Trainer(model, opt, dev, batch=16).fit(n, epochs, step_fn, after_epoch=ctrl.end_epoch)
"""
from __future__ import annotations

import math

import torch


class EarlyStop:
    """Val-metric patience with best-weight restore — one optimized run instead of a fixed epoch budget.
    `val_fn()` returns a higher-is-better validation metric each epoch; training stops after `patience`
    epochs with no improvement and the best-seen weights are restored (so the reported model is the peak,
    not the over-fit tail). The measured alternative to guessing `epochs` — and to retraining N times."""

    def __init__(self, val_fn, patience: int = 10):
        self.val_fn = val_fn
        self.patience = patience
        self.best = -math.inf
        self.best_state = None
        self.since = 0

    def step(self, model) -> bool:
        """Score the epoch; snapshot on improvement. -> True when patience is exhausted (stop now)."""
        v = float(self.val_fn())
        if v > self.best:
            self.best, self.since = v, 0
            self.best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            self.since += 1
        return self.since >= self.patience

    def restore(self, model):
        """Load the best-seen weights back into the model (the peak, not the last epoch)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    def __init__(self, model, opt, device, *, batch: int = 16):
        self.model = model
        self.opt = opt
        self.device = device
        self.batch = batch

    def fit(self, n: int, epochs: int, step_fn, *, after_epoch=None, stop=None):
        """`epochs` passes over `n` items in shuffled minibatches (an UPPER bound when `stop` is set);
        `step_fn(idx)` returns the loss tensor to backprop. `after_epoch()` fires once per epoch (e.g. a
        curriculum's end_epoch). `stop` (an EarlyStop) ends early on val-metric patience + restores best."""
        for _ in range(epochs):
            self.model.train()
            order = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch):
                loss = step_fn(order[i:i + self.batch])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
            if after_epoch is not None:
                after_epoch()
            if stop is not None and stop.step(self.model):
                break
        if stop is not None:
            stop.restore(self.model)
        return self.model
