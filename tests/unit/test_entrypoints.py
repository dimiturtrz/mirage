"""Each CLI front-end exposes exactly its expected subcommands and builds its parser cleanly.

One entry point per concern (run / report / viz); this asserts the `Spec` registries carry the right
commands and construct without error. `run`/`viz` pull in the torchvision method+render registries, so
they're skipped where torchvision is absent (CI); `report` has no torchvision dependency and always runs.
"""
import pytest

import surfscan.report as report
from surfscan.dispatch import Dispatch


def _cmds(specs):
    return {s.name for s in specs}


def test_report_commands():
    assert _cmds(report.REPORTS) == {"evaluate", "results", "sync", "modelcard", "triad-summary"}
    Dispatch.build_parser("surfscan.report", report.REPORTS)


def test_run_commands():
    pytest.importorskip("torchvision")
    from surfscan.experiments.registry import REGISTRY  # noqa: PLC0415
    assert _cmds(REGISTRY) == {"patchcore", "btf", "fused", "featrecon", "draem", "triad", "vae", "pointmae"}
    Dispatch.build_parser("surfscan.run", REGISTRY)


def test_viz_commands():
    pytest.importorskip("torchvision")
    import surfscan.viz as viz  # noqa: PLC0415
    assert _cmds(viz.VIEWS) == {"show", "export"}
    Dispatch.build_parser("surfscan.viz", viz.VIEWS)
