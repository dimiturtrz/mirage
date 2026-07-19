"""Turn a configured method class into a `surfscan.run` subcommand — the generic runner.

A method that satisfies `AnomalyMethod` (fit/score) plus a typed config dataclass needs no bespoke
runner: `MethodCli.method_spec(name, MethodClass, CfgClass)` builds the `Spec` whose args come from the
config schema and whose run constructs the method and feeds it to the harness. The registry becomes data —
adding a single-Method method is one `MethodCli.method_spec(...)` line, no new closures.

`MethodCli.build_method` (namespace + device -> (run_name, method)) is the pure wiring, unit-testable with
a fake method; only the final `Compute.pick_device()` + `harness.run` in the run closure is the
device/mlflow boundary.
"""
from __future__ import annotations

import argparse
from typing import Callable, TypeAlias

import torch

from core.cli_config import CfgT, CliConfig
from core.compute import Compute
from core.method import AnomalyMethod, StateT
from surfscan.dispatch import Dispatch, Spec
from surfscan.evaluation import harness

# A method class as the runner USES it: called with (config, device) to produce a detector. `type[...]`
# would over-constrain (the class object need only be constructible this way), and what it returns is an
# `AnomalyMethod` whose own state type flows through as StateT — so a method's fit/score stay tied.
MethodFactory: TypeAlias = Callable[[CfgT, str | torch.device], AnomalyMethod[StateT]]


class MethodCli:
    """Generic-runner namespace — turn a configured method class into a `surfscan.run` subcommand Spec."""

    @staticmethod
    def build_method(method_cls: MethodFactory[CfgT, StateT], cfg_cls: type[CfgT], args: argparse.Namespace,
                     dev: str | torch.device) -> tuple[str | None, AnomalyMethod[StateT]]:
        """(parsed namespace, device) -> (run_name, method instance). run_name defaults to None when the
        method doesn't declare one (the caller falls back to the subcommand name)."""
        method = method_cls(CliConfig.build_config(cfg_cls, args), dev)
        return getattr(method, "run_name", None), method

    @staticmethod
    def method_spec(name: str, method_cls: MethodFactory[CfgT, StateT], cfg_cls: type[CfgT]) -> Spec:
        def add_args(ap: argparse.ArgumentParser) -> None:
            Dispatch.add_cats(ap)
            CliConfig.add_config_args(ap, cfg_cls)

        def run(args: argparse.Namespace) -> None:  # pragma: no cover  mlflow boundary; build_method is core
            Compute.enable_tf32()
            run_name, method = MethodCli.build_method(method_cls, cfg_cls, args, Compute.pick_device())
            harness.Harness.run(run_name or name, method, cats=args.cats)

        return Spec(name, add_args, run)
