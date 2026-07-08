# mirage

**Personal learning project — 3D surface-defect anomaly detection on MVTec 3D-AD, toward a physics-based synthetic-data engine and robust sim-to-real evaluation. Edge-deployable.**

> Working name `mirage` (synthetic that must survive contact with the real). Repo folder may differ.

![mirage anomaly viewer — the working detector's heatmap on a held-out bagel](docs/media/viewer-anomaly.png)

*The working detector (PatchCore feature memory bank) scoring a held-out bagel — bright = anomalous,
lit at the defect. Rotatable in-browser → [`pointcloud-viewer/`](pointcloud-viewer/). (Scan from
[MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad), CC BY-NC-SA.)*

mirage detects **defects on 3D surface scans**, training only on *good* examples. Stage 0 stands up the
detector + a rigorous eval harness on **real** data and runs a like-for-like comparison of methods; the
differentiator (Stage 1) is a **synthetic-defect generator** + the **sim-to-real gap** measurement the
field lacks.

It's also how I'm ramping into 3D perception: the data-engine + evaluation discipline carry over from
prior acoustic-detection work; the 3D modality I learn as I go. Sibling project, same philosophy,
different domain: [systole](https://github.com/dimiturtrz/cardiac-seg) (cardiac MRI → ejection fraction,
rigorously evaluated).

## Stage 0 result — the measured investigation
Not a leaderboard number — the *measured contrast*. Per-category models, scored through one verified
eval harness (image-AUROC + pixel-AU-PRO). Full table + diagnostics → [`docs/RESULTS.md`](docs/RESULTS.md).

| method | paradigm | pixel AU-PRO |
|---|---|---|
| VAE / inpaint / DRAEM / fused (4+ variants) | reconstruct pixels → error | **0.095** (≈ random) |
| **PatchCore** (the deployable) | **memory bank** — nearest-neighbour to normal features, *no reconstruction* | **0.91** |
| feature-recon | reconstruct *features* → error | 0.91 |
| SOTA (multimodal fusion, *papers*) | RGB + 3D geometry fusion (M3DM 0.96 → DCRDF-Net 0.99) | ~0.96–0.99 |

**The finding:** pixel-reconstruction is ≈ random at localization — *measured* across four variants,
then *diagnosed*: raw-residual tracks geometric **complexity** (a curved rim is hard to rebuild), not
**defect-ness** (a smooth defect rebuilds fine → low residual). What works is **"compare to normal"** —
and two *different* paradigms get there: a **memory bank** (PatchCore: store normal features, score by
nearest-neighbour distance — no rebuilding at all) and **reconstruction moved into feature space**
(feature-recon). Both 0.91. The deployable is the memory bank. The gap to SOTA is named and measured:
multimodal RGB+3D fusion (M3DM through current DCRDF-Net ~0.99), not bank/resolution tuning. The contribution is the eval rigor + the
measured mechanism, the way [systole](https://github.com/dimiturtrz/cardiac-seg) reported its triad.

## The hard question (Stage 1, the differentiator)
The easy demo trains on real defects and reports a flattering number. The real questions: can a model
trained on **synthetic** defects detect **real** ones, *how much does it lose*, and is it **calibrated**
under that shift? Sim-to-real for 3D anomaly is only now being charted — [SiM3D](https://arxiv.org/abs/2506.21549)
(2025) is the first synthetic→real 3D-anomaly benchmark (single-instance, CAD→real). A physics-based
*defect-generation* engine + closed-loop curriculum + a measured gap on commodity MVTec-style
scans is still open ground — that's the contribution.

## Data
**MVTec 3D-AD** (Bergmann et al., VISAPP 2022) — real industrial 3D anomaly benchmark, 10 categories
scanned with a Zivid structured-light sensor (per-pixel rgb + xyz position map + defect masks). Lives
**outside the repo** (licensing + size): set one root in `paths.yaml`, drop the download under
`<root>/raw/mvtec_3d_anomaly_detection/`. CC BY-NC-SA 4.0 — non-commercial, attribution required; not
redistributed here.

## Layout
`core/` (reusable engine) + `surfscan/` (the ML science) + `sim/` (the synthetic engine, own env):
```
core/                  the reusable, method-agnostic engine
  config.py            data root from paths.yaml (-> raw/ + processed/ + synth/)
  data/                adapters (mvtec · synth) + preprocess + unified store
  method.py            the (fit_fn, score_fn) contract the harness scores
surfscan/              the science layer
  models/              vae · inpaint · draem · feat_recon · patchcore · fpfh_bank (BTF)
  training/            train spine + hparams + losses
  experiments/         run_* — per-method benchmark runners (produce the AU-PRO board)
  evaluation/          harness (spine) · metrics · scoring · diagnostics · sync_numbers
  visualization/       show.py (3D viewer) + export_web.py (web-viewer data)
sim/                   synthetic-defect engine — Isaac/Replicator (its own env)
docs/                  PLAN.md · RESULTS.md (+ RESULTS.json canonical) · STRUCTURE.md
learning/ research/ tests/ (unit + integration) · pointcloud-viewer/
```

## Quickstart
```bash
uv sync --extra features          # .venv + deps; torch/torchvision cu130 (Blackwell/RTX 5090) auto-resolved
cp paths.example.yaml paths.yaml          # set `data:` to your data root
uv run python -m core.data.store        # consolidate raw -> processed (needs the dataset)

# the working detector, scored through the eval harness (logs params/metrics to MLflow):
uv run python -m surfscan.run patchcore   # image-AUROC + AU-PRO + per-defect, all 10 categories
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare every method run in the browser

# see it: the defect glows under the working detector (vs the VAE's backwards residual):
uv run python -m surfscan.viz show --cat bagel --split test --defect hole --idx 0 --processed --patchcore
```
Experiment tracking is **MLflow** (canonical store: `mlflow.db` + `mlartifacts/`, gitignored) — every
method/training run logs params, metrics (per-category + per-defect), the aggregate, and trained models.

## How it's built
Agent-driven build, human-owned judgment — the modeling, measurement correctness, and evaluation are
mine; coding agents scaffold the plumbing. Data-structure reasoning and evaluation discipline carry over
from prior ML work; the 3D specifics I learn as I go ([`learning/`](learning/)).

## License
Code: see [LICENSE](LICENSE). The dataset is **not** included and carries its own license
(CC BY-NC-SA 4.0) — obtain it from [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).
