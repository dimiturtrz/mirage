"""Opt-in Isaac boot smoke — needs a GPU + the kit runtime, so it SKIPS unless explicitly enabled.
Covers the boot path (kit kernel + USD stage) AND the pxr build_mesh staging that the pure unit tests
can't reach. Run on the Isaac machine:

    cd sim && OMNI_KIT_ACCEPT_EULA=YES SIM_ISAAC_BOOT=1 uv run pytest tests/test_boot_smoke.py -s

(twin_geom pure tests need no boot:  cd sim && uv run pytest tests/unit)
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SIM_ISAAC_BOOT") != "1",
    reason="Isaac boot smoke is opt-in (needs GPU + kit runtime); set SIM_ISAAC_BOOT=1 to run")


def test_isaac_boot_and_build_mesh():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import omni.usd

        from twin_obj import build_mesh
        stage = omni.usd.get_context().get_stage()
        assert stage is not None                                   # kit kernel + USD stage up
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        faces = np.array([[0, 1, 2]], np.int32)
        mesh = build_mesh(stage, verts, faces)
        assert len(mesh.GetPointsAttr().Get()) == 3                # build_mesh staged the points
        print("BOOT_SMOKE_OK", flush=True)                        # sentinel — kit fastShutdown eats the summary
    finally:
        app.close()
