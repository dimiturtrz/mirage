"""Isaac boot smoke — boots the kit kernel, opens a USD stage, and stages a mesh via build_mesh (the
pxr path the pure twin_geom tests can't reach). Runs by DEFAULT in the sim env (GPU + isaacsim present);
skips only where isaacsim isn't installed.

The boot runs in a SUBPROCESS: SimulationApp shuts down with fastShutdown (os._exit), which would kill
the whole pytest session if booted in-process — the subprocess contains that hard-exit, so this stays a
normal, default-collected test and pytest reporting survives. ~25s (kit boot).

    cd sim && OMNI_KIT_ACCEPT_EULA=YES uv run pytest tests/test_boot_smoke.py -s
"""
import importlib.util
import os
import subprocess
import sys

import pytest

_SIM_DIR = os.path.dirname(os.path.dirname(__file__))

_BOOT = """
import numpy as np
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import omni.usd

    from twin_obj import build_mesh
    stage = omni.usd.get_context().get_stage()
    assert stage is not None                                       # kit kernel + USD stage up
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    mesh = build_mesh(stage, verts, np.array([[0, 1, 2]], np.int32))
    assert len(mesh.GetPointsAttr().Get()) == 3                    # build_mesh staged the points
    print("BOOT_SMOKE_OK", flush=True)
finally:
    app.close()
"""


@pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None,
                    reason="isaacsim not installed (boot smoke only runs in the sim env)")
def test_isaac_boot_and_build_mesh():
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    r = subprocess.run([sys.executable, "-c", _BOOT], cwd=_SIM_DIR, env=env,  # noqa: S603
                       capture_output=True, text=True, timeout=300)
    assert "BOOT_SMOKE_OK" in r.stdout, (
        f"kit boot / build_mesh failed (rc={r.returncode})\n"
        f"--- stdout tail ---\n{r.stdout[-2000:]}\n--- stderr tail ---\n{r.stderr[-2000:]}")
