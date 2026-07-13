"""Turn a configured method class into a `surfscan.run` subcommand — the generic runner.

A method that satisfies `AnomalyMethod` (fit/score) plus a typed config dataclass needs no bespoke
runner: `method_spec(name, MethodClass, CfgClass)` builds the `Spec` whose args come from the config
schema and whose run constructs the method and feeds it to the harness. The registry becomes data —
adding a single-Method method is one `method_spec(...)` line, no new closures.

`build_method` (namespace + device -> (run_name, method)) is the pure wiring, unit-testable with a fake
method; only the final `pick_device()` + `harness.run` in the run closure is the device/mlflow boundary.
"""
from __future__ import annotations

from core.cli_config import CliConfig
from core.compute import Compute
from surfscan.dispatch import Spec, add_cats
from surfscan.evaluation import harness


def build_method(method_cls, cfg_cls, args, dev):
    """(parsed namespace, device) -> (run_name, method instance). run_name defaults to None when the
    method doesn't declare one (the caller falls back to the subcommand name)."""
    method = method_cls(CliConfig.build_config(cfg_cls, args), dev)
    return getattr(method, "run_name", None), method


def method_spec(name: str, method_cls, cfg_cls) -> Spec:
    def add_args(ap):
        add_cats(ap)
        CliConfig.add_config_args(ap, cfg_cls)

    def run(args):  # pragma: no cover  device pick + mlflow harness boundary; build_method is the core
        Compute.enable_tf32()
        run_name, method = build_method(method_cls, cfg_cls, args, Compute.pick_device())
        harness.Harness.run(run_name or name, method, cats=args.cats)

    return Spec(name, add_args, run)
