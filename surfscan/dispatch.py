"""One argparse subcommand dispatcher, shared by the three CLI front-ends (`run`, `report`, `viz`).

Fewer entry points = better design: instead of one `__main__` per method/report/view, each of those is
a `Spec` (name + how to register its args + what to run), and a single front-end routes to it by name.

    Spec("patchcore", add_args, run)                   # a method/report/view registers itself as a Spec
    Dispatch.dispatch("surfscan.run", REGISTRY, argv)  # the front-end builds subparsers + routes to Spec.run

`Dispatch.build_parser` is pure (specs -> parser) and `Dispatch.route` is pure (specs + parsed args ->
the chosen Spec),
so the wiring is unit-testable without touching argv or running any GPU/IO body.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Spec:
    """A named subcommand: `add_args` registers its options on a subparser, `run` executes it."""
    name: str
    add_args: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], None]


class Dispatch:
    """Argparse subcommand dispatch namespace — build subparsers from specs and route to the chosen one."""

    @staticmethod
    def add_cats(ap: argparse.ArgumentParser) -> None:
        """The `--cats` option shared by every runner (default: all categories)."""
        ap.add_argument("--cats", nargs="*", default=None)

    @staticmethod
    def build_parser(prog: str, specs: Sequence[Spec]) -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(prog=prog)
        sub = ap.add_subparsers(dest="cmd", required=True)
        for s in specs:
            s.add_args(sub.add_parser(s.name))
        return ap

    @staticmethod
    def route(specs: Sequence[Spec], args: argparse.Namespace) -> Spec:
        return {s.name: s for s in specs}[args.cmd]

    @staticmethod
    def dispatch(prog: str, specs: Sequence[Spec], argv=None) -> None:  # pragma: no cover  argv/exec boundary
        args = Dispatch.build_parser(prog, specs).parse_args(argv)
        Dispatch.route(specs, args).run(args)
