"""One SGD loop — shuffle, minibatch, zero/backward/step — shared by every trained method.

The method supplies model + optimizer + a `step_fn(idx) -> loss` tensor that owns batch assembly, the
forward, any autocast, and the loss; the Trainer owns only the epoch/minibatch/optimizer mechanics that
were hand-rolled four times (draem, featrecon, triad, the vae in train.py). Device-injected, no
torchvision — so it's unit-testable on a tiny cpu model.

    Trainer(model, opt, dev, batch=16).fit(n, epochs, step_fn, after_epoch=ctrl.end_epoch)
"""
from __future__ import annotations

import torch


class Trainer:
    def __init__(self, model, opt, device, *, batch: int = 16):
        self.model = model
        self.opt = opt
        self.device = device
        self.batch = batch

    def fit(self, n: int, epochs: int, step_fn, *, after_epoch=None):
        """`epochs` passes over `n` items in shuffled minibatches; `step_fn(idx)` returns the loss
        tensor to backprop. `after_epoch()` fires once per epoch (e.g. a curriculum's end_epoch)."""
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
        return self.model
