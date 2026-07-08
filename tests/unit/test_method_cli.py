"""The generic runner wiring — method class + config -> Spec, with the pure build_method core.

Uses a fake method (no torchvision) so the protocol conformance + arg wiring are tested without a
backbone. build_method is the device/config wiring; method_spec is the Spec factory.
"""
import argparse
from dataclasses import dataclass

from core.method import AnomalyMethod
from surfscan.method_cli import build_method, method_spec


@dataclass(frozen=True)
class FakeCfg:
    coreset: float = 0.1


class FakeMethod:
    run_name = "fake_run"

    def __init__(self, cfg, dev):
        self.cfg = cfg
        self.dev = dev

    def fit(self, cat):
        return cat

    def score(self, state, cat):
        return state


class NoNameMethod:
    def __init__(self, cfg, dev):
        self.cfg = cfg
        self.dev = dev

    def fit(self, cat):
        return cat

    def score(self, state, cat):
        return state


def test_fake_method_satisfies_protocol():
    assert isinstance(FakeMethod(FakeCfg(), "cpu"), AnomalyMethod)


def test_build_method_wires_config_and_device():
    args = argparse.Namespace(coreset=0.3, cats=None)
    run_name, method = build_method(FakeMethod, FakeCfg, args, "cpu")
    assert run_name == "fake_run" and method.cfg.coreset == 0.3 and method.dev == "cpu"


def test_build_method_run_name_optional():
    args = argparse.Namespace(coreset=0.1, cats=None)
    run_name, _ = build_method(NoNameMethod, FakeCfg, args, "cpu")
    assert run_name is None


def test_method_spec_builds_spec_and_args():
    spec = method_spec("fake", FakeMethod, FakeCfg)
    assert spec.name == "fake"
    ap = argparse.ArgumentParser()
    spec.add_args(ap)
    ns = ap.parse_args(["--coreset", "0.5"])
    assert ns.coreset == 0.5 and ns.cats is None
