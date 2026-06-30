"""Isaac Sim headless bootstrap smoke test — does the kit kernel come up on the RTX 5090?

The p3h spike's pass-criterion #1. Run from sim/:
    OMNI_KIT_ACCEPT_EULA=YES uv run python smoke_init.py
"""
from isaacsim import SimulationApp

print("booting SimulationApp (headless)...", flush=True)
app = SimulationApp({"headless": True})
print("KIT KERNEL OK", flush=True)

import omni.usd  # noqa: E402

stage = omni.usd.get_context().get_stage()
print("USD STAGE OK:", stage is not None, flush=True)

app.close()
print("CLOSED OK", flush=True)
