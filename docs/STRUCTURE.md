# mirage — code structure

How the repo is laid out and why. The organizing idea (shared with the siblings): **the eval harness
is the product, not the model** — so the code separates a reusable, method-agnostic evaluation *spine*
from the interchangeable *methods* it scores, and treats synthetic-data generation as a second data
source behind the same adapter boundary.

## Layout
Two packages — a reusable engine and the science on top — plus the sim engine in its own env:
```
core/                the reusable, method-agnostic ENGINE (perception env — uv .venv, py3.12, torch cu130)
  config.py          one data root from paths.yaml -> raw/ + processed/ + synth/
  data/              static/ (adapters mvtec · synth -> unified store; dataset GPU-resident) + dynamic/ (Defects + twin engine)
  method.py          the (fit_fn, score_fn) contract + ScoreArrays the harness scores
surfscan/            the SCIENCE layer (imports core)
  models/            the three method families: vae · inpaint · draem · feat_recon · patchcore · fpfh_bank
  training/          train spine (registry-dispatched) + hparams + losses
  experiments/       run_* — per-method drivers that feed the harness (the AU-PRO board)
  evaluation/        the spine: harness · metrics · scoring · diagnostics · sync_numbers
  tracking.py        MLflow wrapper
  visualization/     3D viewer + web-viewer export
sim/                 the synthetic-defect ENGINE (its OWN env — Isaac/Replicator, py3.11/3.12)
docs/                PLAN · RESULTS.md (+ RESULTS.json canonical numbers) · STRUCTURE (this)
tests/               unit (equivalence-class) + integration (module-pair)
```

## The eval spine + the method contract
The harness is **method-agnostic**: it speaks one contract, `core/method.Method` — a
`(fit_fn, score_fn)` pair returning a named `ScoreArrays` — that **all three anomaly paradigms**
satisfy, so one spine scores them identically (cf. mindscape's `Decoder` contract).

```
                          fit_fn(cat) -> state
                          score_fn(state, cat) -> ScoreArrays
   memory bank  ─┐        (amaps · valids · masks · scores · labels · defects)
   PatchCore/BTF │
   reconstruction├──────► evaluation/harness.aggregate ──► per-category + mean + per-defect
   VAE/feat_recon│           (pure: metrics only)      └─► harness.run ──► MLflow
   synthesis     │
   DRAEM        ─┘
```
`aggregate` is pure (compute, no IO — unit-testable); `run` wraps it with MLflow logging (a fresh
method run, or `run_id=` to log back into an existing training run). `evaluate.py` is just the
recon-model adapter to this same spine.

## The data boundary (real + synthetic = one cloud)
One-root `paths.yaml` → `<data>/{raw,processed,synth}`. Two adapters emit the **same on-disk format**
(rgb · xyz position-map · gt), so the store/preprocess/harness ingest them identically:

```
raw/mvtec…  ──(core/data/static/mvtec.py)──┐
                                           ├──► core/data/static/store ──► dataset (GPU-resident) ──► harness
synth/…     ──(core/data/static/synth.py)──┘        ▲
   ▲                                         │
   └── core/data/dynamic engine (Isaac/Replicator) renders the labeled task-graph
```
Real vs synthetic scored through one harness **is** the sim-to-real gap — the Stage-1 contribution.
The engine runs in the **main** env (no nested venv — bd 53z proved co-resolution); isaacsim just sits
behind the opt-in `sim` extra so gate/CI runs never pull it (see
[`core/data/dynamic/README.md`](../core/data/dynamic/README.md)).

## Numbers live in one place
`docs/RESULTS.json` is the canonical source of published numbers; `evaluation/sync_numbers.py` renders
the `docs/RESULTS.md` tables from it (between `<!-- results:X -->` markers) so metrics can't drift.
The root `README.md` is the deliberate exception — a rich, self-contained portfolio face, not templated.
