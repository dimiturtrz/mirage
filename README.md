# mirage

**Personal learning project — 3D surface-defect anomaly detection on MVTec 3D-AD, toward a physics-based synthetic-data engine and honest sim-to-real evaluation. Edge-deployable.**

> Working name `mirage` (synthetic that must survive contact with the real). Repo folder may differ.

mirage detects **defects on 3D surface scans**, training only on *good* examples. Stage 0 stands up the
detector + a rigorous eval harness on **real** data and runs an honest comparison of methods; the
differentiator (Stage 1) is a **synthetic-defect generator** + the **sim-to-real gap** measurement the
field lacks.

It's also how I'm ramping into 3D perception: the data-engine + evaluation discipline carry over from
prior acoustic-detection work; the 3D modality I learn as I go. Sibling project, same philosophy,
different domain: [systole](https://github.com/dimiturtrz/cardiac-seg) (cardiac MRI → ejection fraction,
honestly evaluated).

## Stage 0 result — the honest investigation
Not a leaderboard number — the *measured contrast*. Per-category models, scored through one verified
eval harness (image-AUROC + pixel-AU-PRO). Full table + diagnostics → [`docs/RESULTS.md`](docs/RESULTS.md).

| method | pixel AU-PRO |
|---|---|
| reconstruction (VAE / inpaint / DRAEM / fused — 4+ variants) | **0.095** (≈ random) |
| **PatchCore / feature-recon** (the working method) | **0.91** |
| SOTA (M3DM, *paper*) | ~0.96 |

**The finding:** reconstruction-residual is ≈ random at localization — *measured* across four variants,
then *diagnosed*: raw-residual tracks geometric **complexity** (a curved rim is hard to rebuild), not
**defect-ness** (a smooth defect rebuilds fine → low residual). Move to **"compare to normal"** (a
feature memory bank) and it localizes — two independent feature methods both land at 0.91. The remaining
gap to SOTA is named and measured: M3DM-scale multimodal geometry, not bank or resolution tuning. The
contribution is the eval rigor + the honest mechanism, the way [systole](https://github.com/dimiturtrz/cardiac-seg)
reported its triad.

## The honest question (Stage 1, the differentiator)
The easy demo trains on real defects and reports a flattering number. The real questions: can a model
trained on **synthetic** defects detect **real** ones, *how much does it lose*, and is it **calibrated**
under that shift? No standardized sim-to-real 3D-anomaly benchmark exists — that gap is the contribution.

## Data
**MVTec 3D-AD** (Bergmann et al., VISAPP 2022) — real industrial 3D anomaly benchmark, 10 categories
scanned with a Zivid structured-light sensor (per-pixel rgb + xyz position map + defect masks). Lives
**outside the repo** (licensing + size): set one root in `paths.yaml`, drop the download under
`<root>/raw/mvtec_3d_anomaly_detection/`. CC BY-NC-SA 4.0 — non-commercial, attribution required; not
redistributed here.

## Layout
```
mirage/mirage/
  config.py            data root from paths.yaml (-> raw/ + processed/)
  data/                adapter (mvtec) + preprocess + unified store (meta.csv = the inventory)
  models/              vae · inpaint · draem · feat_recon · patchcore · fpfh_bank (BTF)
  training/            run_* (per-category fit/score) + train loops
  evaluation/          metrics (AU-PRO/AUROC) · scoring · diagnostics · harness
  visualization/show.py   3D viewer: raw / processed / normals / curvature / anomaly heatmap
docs/                  PLAN.md (staged plan) · RESULTS.md (the findings)
learning/              theory notes (the 3D-geometry ramp)
```

## Quickstart
```bash
conda create -n mirage python=3.11 -y && conda activate mirage
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # Blackwell/RTX 5090
pip install -e .
cp paths.example.yaml paths.yaml          # set `data:` to your data root
python -m mirage.data.store               # consolidate raw -> processed (needs the dataset)

# the working detector, scored through the eval harness:
python -m mirage.training.run_patchcore   # image-AUROC + AU-PRO + per-defect, all 10 categories

# see it: the defect glows under the working detector (vs the VAE's backwards residual):
python -m mirage.visualization.show --cat bagel --split test --defect hole --idx 0 --processed --patchcore
```

## How it's built
Agent-driven build, human-owned judgment — the modeling, measurement correctness, and evaluation are
mine; coding agents scaffold the plumbing. Data-structure reasoning and evaluation discipline carry over
from prior ML work; the 3D specifics I learn as I go ([`learning/`](learning/)).

## License
Code: see [LICENSE](LICENSE). The dataset is **not** included and carries its own license
(CC BY-NC-SA 4.0) — obtain it from [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).
