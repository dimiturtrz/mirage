# sim/ — the shared Isaac Sim env (data-gen + control rollouts)

One isolated Isaac Sim env, two consumers on the shared `core`:

- **Data-gen (Replicator)** — authoring of defect scenes → the labeled task-graph (rgb · depth · normals ·
  semantic/instance seg · anomaly mask), which feeds the perception pipeline as a second data source.
- **Control rollouts (PhysX)** — `isaac_reach.py` is a rigid-body reach env satisfying `core.rollout.Env`,
  used by `isaac_gap.py` to re-measure the control leg's sim-to-real **policy gap** at PhysX fidelity (the
  rung above the numpy point mass). The clean seam: Replicator = data-gen (the engine's old charter),
  PhysX = control.

## Why a separate env
Isaac Sim bundles its own kit runtime and pins deps that **can't co-resolve** with the perception
package's env (py3.12 + torch cu130). So the *env* is isolated — but the *code* lives here in the
repo, and the *output* flows into the same store via [`core/data/synth.py`](../core/data/synth.py).
Standard sim split: engine renders → adapter ingests → harness scores. One repo, two envs, one data contract.

## Setup
```bash
cd sim
uv sync                                   # builds sim/.venv (py3.12, isaacsim 6.0.1.0 from pypi.nvidia.com)
```
Verify the kit kernel boots on the 5090 with the boot smoke (see [Tests](#tests)).
Requires Windows long-path enabled (`LongPathsEnabled=1`). Blackwell note: use a standard `Camera`,
not `TiledCamera` (IsaacLab #4951 hangs on sm_120) — TiledCamera only matters for parallel-env RL.

## Tests
```bash
OMNI_KIT_ACCEPT_EULA=YES uv run pytest tests           # all: pure twin_geom + the Isaac boot smoke (~GPU)
uv run pytest tests/unit                               # just the pure twin_geom helpers — no boot
```
`twin_geom.py` holds the pxr-free geometry (OBJ parse, depth back-projection, the Gaussian defect) so it
unit-tests in any numpy env. The boot smoke (`test_boot_smoke.py`) boots the kit kernel + a USD stage and
stages a mesh via `build_mesh` — the pxr path the pure tests can't reach. It runs by default in the sim
env (skips where isaacsim is absent); the boot runs in a subprocess so SimulationApp's fastShutdown
hard-exit can't kill the pytest session.

## Data handoff (perception)
`generate.py` (this env) writes renders to `<data root>/synth/<category>/...`; the perception env's
`core/data/synth.py` adapter maps them to the common store schema (same shape as the MVTec adapter),
so synthetic and real are one 2-source cloud the harness scores identically — the sim-to-real gap.

## Control rollouts (policy gap at PhysX fidelity)
```bash
OMNI_KIT_ACCEPT_EULA=YES uv run python isaac_gap.py [--seeds 3] [--episodes 40]
```
`isaac_reach.py` is a rigid-sphere reach env (`core.rollout.Env`): zero gravity + PhysX linear damping
`drag/mass` + a planar force `gain·a` reproduce the point-mass equation `acc = (gain·a − drag·v)/mass` at
solver fidelity. `isaac_gap.py` imports the control leg's `BCPolicy` / `PDExpert` / `Rollout` / gap metric
**unchanged** and re-measures the payload sweep — only the `Env` differs. Opt-in (GPU + ~30 s kit boot),
not CI-gated. Result + point-mass comparison → [`interpretations/control/`](../interpretations/control/).
