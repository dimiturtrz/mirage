# sim/ — the Isaac Sim engine (data-gen + control rollouts)

The synthetic-data **engine** leg, two consumers on the shared `core`:

- **`datagen/` (Replicator)** — authoring of defect scenes → the labeled task-graph (rgb · depth · normals ·
  semantic/instance seg · anomaly mask), which feeds the perception pipeline as a second data source.
  `twin_geom` holds the pxr-free geometry (OBJ parse, depth back-projection, the Gaussian defect); `twin_obj`
  stages it into a USD mesh; `generate_twin_synth` / `render_twin` boot the kit runtime and render.
- **`control/` (PhysX)** — `isaac_reach` is a rigid-body reach env satisfying `core.rollout.Env`, used by
  `isaac_gap` to re-measure the control leg's sim-to-real **policy gap** at PhysX fidelity (the rung above the
  numpy point mass). The clean seam: Replicator = data-gen (the engine's charter), PhysX = control.

## One env
Isaac Sim runs in the **main** uv env — no nested venv. [bd 53z](../.beads/issues.jsonl) proved co-resolution:
isaacsim 6.0.1 declares no torch pin, so torch stays cu130 (the RTX 5090 build), the perception deps
(open3d/timm/onnx/torchvision) import alongside it, and the kit kernel boots + stages a mesh in the same
site-packages. isaacsim ships cp312 wheels only (hence the repo's 3.12 floor) and pulls ~156 isaacsim/omni
packages from `pypi.nvidia.com`, so it sits behind the optional **`sim` extra** — base perception devs never
install it. The sim-to-real split is a *data contract* ([`core/data/synth.py`](../core/data/synth.py) adapts
the engine's output into the common store), not an env boundary: engine renders → adapter ingests → harness
scores. One repo, one env, one data contract.

## Setup
```bash
uv sync --extra sim                       # adds isaacsim 6.0.1.0 (pypi.nvidia.com) to the main env
```
Requires Windows long-path enabled (`LongPathsEnabled=1`). Blackwell note: use a standard `Camera`,
not `TiledCamera` (IsaacLab #4951 hangs on sm_120) — TiledCamera only matters for parallel-env RL.

## Data-gen
```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m sim.datagen.generate_twin_synth --views 8
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m sim.datagen.render_twin --cat bagel
```
The engine writes renders to `<data root>/synth/<category>/...` — an rgb png plus an organized camera-frame
xyz position map (tiff), back-projected from depth (the same shape as the real MVTec scans). The perception
env's [`core/data/synth.py`](../core/data/synth.py) adapter maps them to the common store schema (same shape
as the MVTec adapter), so synthetic and real are one 2-source cloud the harness scores identically — the
sim-to-real gap.

## Tests
The sim tests live in the main tree (they mirror the package): the pure `twin_geom` helpers under
[`tests/unit/sim/datagen/`](../tests/unit/sim/datagen/) run in any numpy env, and the Isaac boot smoke under
[`tests/integration/sim/`](../tests/integration/sim/) boots the kit kernel + a USD stage and stages a mesh via
`build_mesh` — the pxr path the pure tests can't reach.
```bash
uv run pytest tests/unit/sim                              # pure geometry — no boot, no isaacsim
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim pytest tests/integration/sim   # + the boot smoke (~GPU)
```
The boot smoke skips where isaacsim is absent (base suite stays green) and boots in a subprocess so
SimulationApp's fastShutdown hard-exit can't kill the pytest session.

## Control rollouts (policy gap at PhysX fidelity)
```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --extra sim python -m sim.control.isaac_gap [--seeds 3] [--episodes 40]
```
`isaac_reach` is a rigid-sphere reach env (`core.rollout.Env`): zero gravity + PhysX linear damping
`drag/mass` + a planar force `gain·a` reproduce the point-mass equation `acc = (gain·a − drag·v)/mass` at
solver fidelity. `isaac_gap` imports the control leg's `BCPolicy` / `PDExpert` / `Rollout` / gap metric
**unchanged** and re-measures the payload sweep — only the `Env` differs. Opt-in (GPU + ~30 s kit boot),
not CI-gated. Result + point-mass comparison → [`interpretations/control/`](../interpretations/control/).
