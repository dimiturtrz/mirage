"""Deploy cost-model entry point — profile model footprints, project onto accelerators.

    python -m surfscan.deploy profile     # measure params/FLOPs/act-mem/disk -> docs/DEPLOY_COST.json

Each step is a `Spec` in its module; this front-end builds subparsers and routes. See also
`surfscan.run` (experiments) and `surfscan.report` (docs/metrics).
"""
from __future__ import annotations

from surfscan.deploy import profile
from surfscan.dispatch import Dispatch

STEPS = [profile.SPEC]

if __name__ == "__main__":
    Dispatch.dispatch("surfscan.deploy", STEPS)
