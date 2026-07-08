"""The shared SGD loop — batch/epoch bookkeeping + it actually learns, on a tiny cpu model.

Trainer owns shuffle+minibatch+opt; the step_fn owns the forward+loss. Verified: it calls step_fn once
per minibatch (ceil(n/batch) per epoch), after_epoch once per epoch, and drives the loss down.
"""
import torch

from surfscan.training.trainer import Trainer


def test_calls_step_per_minibatch_and_after_each_epoch():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    x, y = torch.randn(10, 4), torch.randn(10, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    calls = {"step": 0, "epoch": 0}

    def step(idx):
        calls["step"] += 1
        return ((model(x[idx]) - y[idx]) ** 2).mean()

    Trainer(model, opt, "cpu", batch=4).fit(
        10, epochs=3, step_fn=step, after_epoch=lambda: calls.__setitem__("epoch", calls["epoch"] + 1))
    assert calls["epoch"] == 3
    assert calls["step"] == 3 * 3          # ceil(10/4) = 3 minibatches per epoch


def test_learns_a_linear_map():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    x = torch.randn(64, 4)
    y = x @ torch.randn(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []

    def step(idx):
        return ((model(x[idx]) - y[idx]) ** 2).mean()

    Trainer(model, opt, "cpu", batch=16).fit(
        64, epochs=25, step_fn=step,
        after_epoch=lambda: losses.append(float(((model(x) - y) ** 2).mean().detach())))
    assert losses[-1] < losses[0]          # the loop drives the loss down
