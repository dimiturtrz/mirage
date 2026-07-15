"""Deploy cost-model entry point — profile model footprints, project onto accelerators.

    python -m surfscan.deploy profile       # measure params/FLOPs/act-mem/disk -> docs/deployment/MODEL_FOOTPRINT.json
    python -m surfscan.deploy accelerators  # cited edge-accelerator spec table -> docs/deployment/ACCELERATOR_SPECS.json
    python -m surfscan.deploy bank          # PatchCore bank-memory model -> docs/deployment/BANK_MEMORY.json
    python -m surfscan.deploy project       # join model x accelerator -> docs/deployment/DEPLOY_PROJECTION.json

Each step is a `Spec` in its module; this front-end builds subparsers and routes. See also
`surfscan.run` (experiments) and `surfscan.report` (docs/metrics).
"""
from __future__ import annotations

from surfscan.deploy import accelerators, bank, profile, projection
from surfscan.dispatch import Dispatch

STEPS = [profile.SPEC, accelerators.SPEC, bank.SPEC, projection.SPEC]

if __name__ == "__main__":
    Dispatch.dispatch("surfscan.deploy", STEPS)
