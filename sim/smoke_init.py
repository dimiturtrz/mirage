"""Isaac Sim headless bootstrap smoke test — does the kit kernel come up on the RTX 5090?

The p3h spike's pass-criterion #1. Run from sim/:
    OMNI_KIT_ACCEPT_EULA=YES uv run python smoke_init.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> import core.obs
from isaacsim import SimulationApp

from core.obs import Obs

log = Obs.get()
log.info("booting SimulationApp (headless)...")
app = SimulationApp({"headless": True})
log.info("KIT KERNEL OK")

import omni.usd

stage = omni.usd.get_context().get_stage()
log.info("USD STAGE OK: %s", stage is not None)

app.close()
log.info("CLOSED OK")
