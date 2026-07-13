"""Unit test for the training spine on CPU — device is injected, so the GPU-resident loop runs on cpu.

The two external boundaries (load_split = disk, tracking = mlflow) are stubbed; everything else is the
real loop: registry dispatch, epoch/batch slicing, the vae + inpaint steps, loss accumulation. One
epoch on a 4-sample fake split proves the spine end-to-end without a GPU or the store.
"""
from contextlib import contextmanager

import torch

from surfscan.training import train as train_mod
from surfscan.training.hparams import HParams


class _FakeSplit:
    def __init__(self, n=4, ch=3, size=32):
        self.x = torch.randn(n, ch, size, size)
        self.valid = torch.ones(n, 1, size, size)

    @property
    def in_ch(self):
        return self.x.shape[1]

    def __len__(self):
        return self.x.shape[0]


class _StubTracker:
    def __init__(self, metric_steps):
        self.metric_steps = metric_steps

    @contextmanager
    def run(self, *_a, **_k):
        yield "run-xyz"

    def metrics(self, row, step=None):
        self.metric_steps.append((step, row))

    def artifact_json(self, *_a, **_k):
        pass

    def log_model(self, *_a, **_k):
        pass


class _StubTracking:
    def __init__(self):
        self.metric_steps = []
        self.Tracker = _StubTracker(self.metric_steps)


def _patch(monkeypatch, size=32):
    monkeypatch.setattr(train_mod.GpuSplit, "load_split", lambda **_k: _FakeSplit(size=size))
    stub = _StubTracking()
    monkeypatch.setattr(train_mod, "tracking", stub)
    return stub


def _hp(model_type):
    return HParams(model_type=model_type, size=32, base=8, latent=16, depth=3,
                   batch=2, epochs=1, compile=False, bf16=False, grid=4)


def test_train_vae_cpu(monkeypatch):
    stub = _patch(monkeypatch)
    run_id = train_mod.TrainRun.train(_hp("vae"), device="cpu")
    assert run_id == "run-xyz"
    assert stub.metric_steps and stub.metric_steps[0][1]["loss"] == stub.metric_steps[0][1]["loss"]  # not NaN


def test_train_inpaint_cpu(monkeypatch):
    _patch(monkeypatch)
    assert train_mod.TrainRun.train(_hp("inpaint"), device="cpu") == "run-xyz"
