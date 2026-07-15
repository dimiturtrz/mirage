"""Deploy cost-model entry point — profile model footprints, project onto accelerators.

`project` is the deliverable: it measures the models, computes the bank, loads the substrate specs, and
writes the SINGLE deploy document (docs/deployment.json) — model costs + bank + substrates + the verdict
matrix, all in one file. The other three are log-only inspection views of the same underlying data.

    python -m surfscan.deploy project       # THE artifact: everything -> docs/deployment.json
    python -m surfscan.deploy profile       # inspect: measured per-model footprints (logged)
    python -m surfscan.deploy accelerators  # inspect: cited substrate specs (logged)
    python -m surfscan.deploy bank          # inspect: PatchCore bank-memory model (logged)

Each step is a `Spec` in its module; this front-end builds subparsers and routes. See also
`surfscan.run` (experiments) and `surfscan.report` (docs/metrics).
"""
from __future__ import annotations

from surfscan.deploy import accelerators, bank, profile, projection
from surfscan.dispatch import Dispatch

STEPS = [profile.SPEC, accelerators.SPEC, bank.SPEC, projection.SPEC]

if __name__ == "__main__":
    Dispatch.dispatch("surfscan.deploy", STEPS)
