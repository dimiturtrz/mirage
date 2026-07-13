"""The single visualization entry point — inspect a sample or export it to the web viewer.

    python -m surfscan.viz show --cat bagel --split test --defect hole --idx 0 --mask
    python -m surfscan.viz export --samples bagel:test:hole:0 cookie:test:crack:0

Each view is a `Spec` in its module; this front-end builds subparsers and routes. See also
`surfscan.run` (experiments) and `surfscan.report` (docs/metrics).
"""
from __future__ import annotations

from surfscan.dispatch import Dispatch
from surfscan.visualization import export_web, show

VIEWS = [show.SPEC, export_web.SPEC]

if __name__ == "__main__":
    Dispatch.dispatch("surfscan.viz", VIEWS)
