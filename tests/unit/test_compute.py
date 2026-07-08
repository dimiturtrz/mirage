"""Unit tests for the device policy — the single home the root uses to pick + inject a device.

pick_device honours an explicit cpu preference and never claims cuda that isn't there; autocast is a
real context on cuda and a no-op elsewhere (so one code path runs GPU-resident or on an edge CPU).
"""
import torch

from core.compute import autocast, enable_tf32, pick_device


def test_pick_device_cpu_preference():
    assert pick_device("cpu") == "cpu"


def test_pick_device_cuda_falls_back_when_absent():
    dev = pick_device("cuda")
    assert dev == ("cuda" if torch.cuda.is_available() else "cpu")


def test_autocast_noop_on_cpu_tensor():
    x = torch.zeros(1)
    with autocast(x):                       # cpu -> disabled, must not raise
        y = x + 1
    assert y.item() == 1.0


def test_autocast_accepts_device_string():
    with autocast("cpu", amp=True):         # disabled on cpu regardless of amp
        assert torch.zeros(2).sum().item() == 0.0


def test_enable_tf32_sets_high_matmul_precision():
    enable_tf32()
    assert torch.get_float32_matmul_precision() == "high"
