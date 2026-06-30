# sim/ — the synthetic-defect engine (mirage Stage 1)

Isaac Sim / Replicator authoring of defect scenes → the labeled task-graph (rgb · depth · normals ·
semantic/instance seg · anomaly mask), which feeds the perception pipeline as a second data source.

## Why a separate env
Isaac Sim bundles its own kit runtime and pins deps that **can't co-resolve** with the perception
package's env (py3.12 + torch cu130). So the *env* is isolated — but the *code* lives here in the
repo, and the *output* flows into the same store via [`surfscan/data/synth.py`](../surfscan/data/synth.py).
Standard sim split: engine renders → adapter ingests → harness scores. One repo, two envs, one data contract.

## Setup
```bash
cd sim
uv sync                                   # builds sim/.venv (py3.11, isaacsim 5.1.0 from pypi.nvidia.com)
OMNI_KIT_ACCEPT_EULA=YES uv run python smoke_init.py   # p3h spike: kit kernel boots on the 5090?
```
Requires Windows long-path enabled (`LongPathsEnabled=1`). Blackwell note: use a standard `Camera`,
not `TiledCamera` (IsaacLab #4951 hangs on sm_120) — TiledCamera only matters for parallel-env RL.

## Data handoff
`generate.py` (this env) writes renders to `<data root>/synth/<category>/...`; the perception env's
`surfscan/data/synth.py` adapter maps them to the common store schema (same shape as the MVTec adapter),
so synthetic and real are one 2-source cloud the harness scores identically — the sim-to-real gap.
