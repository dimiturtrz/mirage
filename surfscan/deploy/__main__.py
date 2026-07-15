"""Deploy fit model entry point — profile detector footprints, cross them against typed accelerators.

Two artifacts land in the browsable top-level `deploy/` piece:

    python -m surfscan.deploy profile       # measure -> deploy/models_params.json (components + op-classes + bank)
    python -m surfscan.deploy fit           # cross    -> deploy/fit_matrix.json (detector x accelerator x options)

The other two steps are log-only inspection views of the same data:

    python -m surfscan.deploy accelerators  # the loaded, cited typed accelerator specs
    python -m surfscan.deploy bank          # the PatchCore bank-memory model

The accelerator specs themselves are hand-authored source in deploy/accelerators/<type>_params.json.
See also `surfscan.run` (experiments) and `surfscan.report` (docs/metrics).
"""
from __future__ import annotations

from surfscan.deploy import accelerators, bank, fit, profile
from surfscan.dispatch import Dispatch

STEPS = [profile.SPEC, fit.SPEC, accelerators.SPEC, bank.SPEC]

if __name__ == "__main__":
    Dispatch.dispatch("surfscan.deploy", STEPS)
