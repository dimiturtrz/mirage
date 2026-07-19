"""One SGD loop — shuffle, minibatch, zero/backward/step — shared by every trained method.

The method supplies model + optimizer + a `step_fn(idx) -> loss` tensor that owns batch assembly, the
forward, any autocast, and the loss; the Trainer owns only the epoch/minibatch/optimizer mechanics that
were hand-rolled four times (draem, featrecon, triad, the vae in train.py). Device-injected, no
torchvision — so it's unit-testable on a tiny cpu model.

    Trainer(model, opt, dev, batch=16).fit(n, epochs, step_fn, after_epoch=ctrl.end_epoch)
"""
from __future__ import annotations

import math
from collections.abc import Callable
from time import perf_counter

import torch
from torch import Tensor, nn, optim

from core.obs import Obs

log = Obs.get()


class Telemetry:
    """Per-epoch training telemetry — so a bad run shows itself in the first epochs instead of after a
    24-minute blind wait. Emits loss, optional val metric, epoch wall-time + imgs/s to stdout, and the
    early-stop event (stopped-epoch, best-val, restored). Injected into `Trainer`, so every method that
    uses the shared loop gets it free; pass a `sink(dict, step)` (e.g. Tracker.metrics) to also log
    step-metrics to MLflow. `Telemetry.off()` is the silent no-op for tests/tight loops."""

    def __init__(self, tag: str = "train",
                 sink: Callable[[dict[str, float], int], None] | None = None):
        self.tag = tag
        self.sink = sink

    @staticmethod
    def off():
        return Telemetry(sink=None, tag="")

    def epoch(self, i: int, n: int, loss: float, val: float | None,
              secs: float) -> None:
        rate = n / secs if secs > 0 else 0.0
        if self.tag:
            v = f"  val {val:.4f}" if val is not None else ""
            log.info(f"[{self.tag}] epoch {i:3d}  loss {loss:.4f}{v}  {secs:.2f}s  {rate:.0f} img/s")
        if self.sink is not None:
            row = {f"{self.tag}.loss": loss, f"{self.tag}.epoch_s": secs}
            if val is not None:
                row[f"{self.tag}.val"] = val
            self.sink(row, step=i)

    def stopped(self, i: int, best: float) -> None:
        if self.tag:
            log.info(f"[{self.tag}] early-stop @ epoch {i}  best-val {best:.4f}  (best weights restored)")


class EarlyStop:
    """Val-metric patience with best-weight restore — one optimized run instead of a fixed epoch budget.
    `val_fn()` returns a higher-is-better validation metric each epoch; training stops after `patience`
    epochs with no improvement and the best-seen weights are restored (so the reported model is the peak,
    not the over-fit tail). The measured alternative to guessing `epochs` — and to retraining N times."""

    def __init__(self, val_fn: Callable[[], float], patience: int = 10):
        self.val_fn = val_fn
        self.patience = patience
        self.best = -math.inf
        self.best_state: dict[str, Tensor] | None = None
        self.since = 0
        self.last: float | None = None            # most recent val (for telemetry — set each step)

    def step(self, model: nn.Module) -> bool:
        """Score the epoch; snapshot on improvement. -> True when patience is exhausted (stop now)."""
        v = float(self.val_fn())
        self.last = v
        if v > self.best:
            self.best, self.since = v, 0
            self.best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            self.since += 1
        return self.since >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load the best-seen weights back into the model (the peak, not the last epoch)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    def __init__(self, model: nn.Module, opt: optim.Optimizer, device: str, *,
                 batch: int = 16, telem: Telemetry | None = None):
        self.model = model
        self.opt = opt
        self.device = device
        self.batch = batch
        self.telem = telem if telem is not None else Telemetry()

    def fit(self, n: int, epochs: int, step_fn: Callable[[Tensor], Tensor], *,
            after_epoch: Callable[[], None] | None = None,
            stop: EarlyStop | None = None) -> nn.Module:
        """`epochs` passes over `n` items in shuffled minibatches (an UPPER bound when `stop` is set);
        `step_fn(idx)` returns the loss tensor to backprop. `after_epoch()` fires once per epoch (e.g. a
        curriculum's end_epoch). `stop` (an EarlyStop) ends early on val-metric patience + restores best.
        Per-epoch telemetry (loss / val / wall-time) is emitted via `self.telem`."""
        for ep in range(epochs):
            t0 = perf_counter()
            self.model.train()
            order = torch.randperm(n, device=self.device)
            total, nb = 0.0, 0
            for i in range(0, n, self.batch):
                loss = step_fn(order[i:i + self.batch])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                total += float(loss.detach()); nb += 1
            if after_epoch is not None:
                after_epoch()
            done = stop is not None and stop.step(self.model)
            self.telem.epoch(ep, n, total / max(nb, 1), (stop.last if stop is not None else None),
                             perf_counter() - t0)
            if done:
                self.telem.stopped(ep, stop.best)
                break
        if stop is not None:
            stop.restore(self.model)
        return self.model
