"""The method registry — every anomaly method registers one `Spec`; `surfscan.run` routes to it.

Adding a method = write its runner module with a `SPEC`, then list it here. No new entry point, no new
`__main__` — the single `surfscan.run` front-end grows a subcommand automatically.
"""
from __future__ import annotations

from surfscan.dispatch import Spec
from surfscan.experiments import run_all as run_vae
from surfscan.experiments import (
    run_btf,
    run_draem,
    run_featrecon,
    run_fused,
    run_patchcore,
    run_triad,
)

REGISTRY: list[Spec] = [
    run_patchcore.SPEC,
    run_btf.SPEC,
    run_fused.SPEC,
    run_featrecon.SPEC,
    run_draem.SPEC,
    run_triad.SPEC,
    run_vae.SPEC,
]
