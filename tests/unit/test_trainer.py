"""The shared SGD loop — batch/epoch bookkeeping + it actually learns, on a tiny cpu model.

Trainer owns shuffle+minibatch+opt; the step_fn owns the forward+loss. Verified: it calls step_fn once
per minibatch (ceil(n/batch) per epoch), after_epoch once per epoch, and drives the loss down.
"""
import torch

from surfscan.training.trainer import EarlyStop, Trainer


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


def test_early_stop_halts_on_patience():
    # a val metric that stops improving after epoch 3 -> stop fires at 3 + patience, well before epochs
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    x = torch.randn(10, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)         # frozen -> val never improves
    seen = {"n": 0}
    vals = iter([0.5, 0.6, 0.7] + [0.6] * 50)                 # rises then plateaus

    def val_fn():
        seen["n"] += 1
        return next(vals)

    stop = EarlyStop(val_fn, patience=2)
    Trainer(model, opt, "cpu", batch=4).fit(10, epochs=50, step_fn=lambda idx: (model(x[idx]) ** 2).mean(),
                                            stop=stop)
    assert seen["n"] == 5                                     # epochs 1-3 improve, 4-5 stall -> stop at 5
    assert stop.best == 0.7


def test_early_stop_restores_best_weights():
    # the model returned is the PEAK-val snapshot, not the (worse, drifted) last epoch
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)

    def drift():                                             # after_epoch: weights walk away each epoch
        with torch.no_grad():
            model.weight += 1.0

    vals = iter([0.9, 0.3, 0.2, 0.1])                        # peaks at epoch 1, then only worsens
    stop = EarlyStop(lambda: next(vals), patience=2)
    Trainer(model, opt=torch.optim.SGD(model.parameters(), lr=0.0), device="cpu").fit(
        4, epochs=10, step_fn=lambda idx: (model(torch.zeros(1, 2)) ** 2).mean(),
        after_epoch=drift, stop=stop)
    assert float(model.weight.detach().sum()) == 2.0         # epoch-1 snapshot (weight=1 each), not the drift
