"""Isaac boot smoke — boots the kit kernel, opens a USD stage, and stages a mesh via build_mesh (the
pxr path the pure twin_geom tests can't reach). Runs by DEFAULT when isaacsim is present (GPU + the `sim`
extra); skips otherwise, so the base perception suite stays green without the engine installed.

The boot runs in a SUBPROCESS: SimulationApp shuts down with fastShutdown (os._exit), which would kill
the whole pytest session if booted in-process — the subprocess contains that hard-exit, so this stays a
normal, default-collected test and pytest reporting survives. ~25s (kit boot).

    OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim pytest tests/integration/core/data/dynamic/test_boot_smoke.py -s
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]

_BOOT = """
import numpy as np
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import omni.usd

    from core.data.dynamic.twin_obj import TwinObj
    stage = omni.usd.get_context().get_stage()
    assert stage is not None                                       # kit kernel + USD stage up
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    mesh = TwinObj.build_mesh(stage, verts, np.array([[0, 1, 2]], np.int32))
    assert len(mesh.GetPointsAttr().Get()) == 3                    # build_mesh staged the points
    print("BOOT_SMOKE_OK", flush=True)
finally:
    app.close()
"""


@pytest.mark.skipif(importlib.util.find_spec("isaacsim") is None,
                    reason="isaacsim not installed (boot smoke only runs in the sim env)")
def test_isaac_boot_and_build_mesh():
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    r = subprocess.run([sys.executable, "-c", _BOOT], cwd=_REPO_ROOT, env=env,  # noqa: S603
                       capture_output=True, text=True, timeout=300)
    assert "BOOT_SMOKE_OK" in r.stdout, (
        f"kit boot / build_mesh failed (rc={r.returncode})\n"
        f"--- stdout tail ---\n{r.stdout[-2000:]}\n--- stderr tail ---\n{r.stderr[-2000:]}")
