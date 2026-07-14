"""The CLI dispatch core — a `Spec` registers its args + run, one front-end builds subparsers and routes.

This is the pure wiring behind `surfscan.run` / `report` / `viz`: build_parser (specs -> parser) and
route (specs + parsed args -> the chosen Spec) are tested here with fake specs, so the routing is
verified without importing any GPU/torchvision method body.
"""
import argparse

import pytest

from surfscan.dispatch import Dispatch, Spec


def _spec(name, sink):
    return Spec(name,
                lambda ap: ap.add_argument("--x", default="d"),
                lambda a: sink.append((name, a.x)))


def test_add_cats_default_and_list():
    ap = argparse.ArgumentParser()
    Dispatch.add_cats(ap)
    assert ap.parse_args([]).cats is None
    assert ap.parse_args(["--cats", "a", "b"]).cats == ["a", "b"]


def test_build_parser_routes_to_subcommand():
    ns = Dispatch.build_parser("prog", [_spec("alpha", []), _spec("beta", [])]).parse_args(["alpha", "--x", "7"])
    assert ns.cmd == "alpha" and ns.x == "7"


def test_route_selects_named_spec_and_runs():
    sink = []
    specs = [_spec("alpha", sink), _spec("beta", sink)]
    ns = Dispatch.build_parser("prog", specs).parse_args(["beta"])
    Dispatch.route(specs, ns).run(ns)
    assert sink == [("beta", "d")]              # beta's run fired, alpha's did not


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        Dispatch.build_parser("prog", [_spec("alpha", [])]).parse_args([])
