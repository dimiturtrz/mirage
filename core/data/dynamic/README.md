# core/data/dynamic — the twin synthetic-defect engine

The **dynamic** half of `core/data` (static = the real MVTec/Zivid scans; dynamic = generated): a
physics-based generator that renders reconstructed object *twins* into MVTec-format defect scans via Isaac
Sim / Replicator. Split library vs runtime so the pure part stays core-grade and testable:

- **`twin_geom.py` (`TwinGeom`)** — the pxr-free geometry library: OBJ parse, z-depth back-projection to an
  organized camera-frame xyz map, and the physical Gaussian-patch defect (dent/bump). Pure numpy, unit-tested
  ([`tests/unit/core/data/dynamic/`](../../../tests/unit/core/data/dynamic/)) — no kit boot.
- **`twin_obj.py` (`TwinObj`)** — stages parsed `(verts, faces)` into a `UsdGeom.Mesh` (the pxr import is
  local to the method, so the module imports without the `sim` extra).
- **`generate_twin_synth.py` (`TwinSynth`)** / **`render_twin.py` (`TwinRenderQC`)** — the boot-driven render
  **entrypoints**: `main` boots `SimulationApp`, then the class drives Replicator. Isaac/omni imports are
  local to those methods (module imports clean; only *running* needs the extra + a GPU).

## One env, one data contract
Isaac Sim runs in the **main** uv env — no nested venv. [bd 53z](../../../.beads/issues.jsonl) proved
co-resolution: isaacsim 6.0.1 declares no torch pin, so torch stays cu130 (the RTX 5090 build), the
perception deps (open3d/timm/onnx/torchvision) import alongside it, and the kit kernel boots + stages a mesh
in the same site-packages. isaacsim ships cp312 wheels only (the repo's 3.12 floor) and pulls ~156
isaacsim/omni packages from `pypi.nvidia.com`, so it sits behind the optional **`sim` extra** — base
perception devs never install it, and CI gate runs never drag it. The sim-to-real split is a *data contract*
([`core/data/static/synth.py`](../static/synth.py) adapts the engine's output into the common store), not an env boundary:
engine renders → adapter ingests → harness scores.

## Setup
```bash
uv sync --extra sim                       # adds isaacsim 6.0.1.0 (pypi.nvidia.com) to the main env
```
Requires Windows long-path enabled (`LongPathsEnabled=1`). Blackwell note: use a standard `Camera`, not
`TiledCamera` (IsaacLab #4951 hangs on sm_120) — TiledCamera only matters for parallel-env RL.

## Generate
```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m core.data.dynamic.generate_twin_synth --views 8
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m core.data.dynamic.render_twin --cat bagel
```
The engine writes renders out-of-repo under `<data>/synth/twin/<category>/...` — an rgb png plus an organized
camera-frame xyz position map (tiff), back-projected from depth (the same shape as the real MVTec scans). The
[`core/data/static/synth.py`](../static/synth.py) adapter maps them to the common store schema (same shape as the MVTec
adapter), so synthetic and real are one 2-source cloud the harness scores identically — the sim-to-real gap.

## Tests
The pure `twin_geom` helpers run in any numpy env; the Isaac boot smoke boots the kit kernel + a USD stage and
stages a mesh via `TwinObj.build_mesh` (the pxr path the pure tests can't reach). It skips where isaacsim is
absent (base suite stays green) and boots in a subprocess so `SimulationApp`'s fastShutdown hard-exit can't
kill the pytest session.
```bash
uv run pytest tests/unit/core/data/dynamic                                   # pure geometry — no boot
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim pytest tests/integration/core/data/dynamic   # + boot smoke (GPU)
```

The PhysX **control** side of the engine (the sim-to-real *policy* gap) lives with its leg —
[`control/sim/`](../../../control/sim/).
