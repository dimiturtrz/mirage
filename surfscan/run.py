"""The single experiment entry point — one CLI over every anomaly method (memory bank, recon, synth).

    python -m surfscan.run patchcore [--coreset 0.1 ...]
    python -m surfscan.run triad [--arms real synth synth_da ...]
    python -m surfscan.run vae [--epochs 100 ...]

Each method is a `Spec` in `surfscan.experiments.registry`; this front-end just builds the subparsers
and routes. See also `surfscan.report` (docs/metrics) and `surfscan.viz` (visualization).
"""
from __future__ import annotations

from surfscan.dispatch import dispatch
from surfscan.experiments.registry import REGISTRY

if __name__ == "__main__":
    dispatch("surfscan.run", REGISTRY)
