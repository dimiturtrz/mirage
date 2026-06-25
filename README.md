# mirage

**Personal learning project — a physics-based synthetic-data engine for 3D perception: generate defect scenes, detect anomalies. Edge-deployable.**

> Working name `mirage` (synthetic that must survive contact with the real). Repo folder may differ.

mirage detects **defects on 3D surface scans** — and asks the question demos skip: how well does a
detector hold up when the defects it trains on are *generated*, not real? The spine is an
**anomaly detector trained only on "good" examples** (a reconstruction model that learns what normal
looks like and flags what it can't rebuild); the differentiator is a **physics-based synthetic-defect
generator** and the **honest sim-to-real evaluation** of the gap between synthetic-trained and real.
Then it's pushed to run on edge.

It's also how I'm ramping into 3D perception: the synthetic-data + reconstruction-anomaly +
honest-evaluation core is carried over from prior acoustic-detection work; the 3D modality I learn as
I go. Sibling project, same philosophy, different domain:
[systole](https://github.com/dimiturtrz/cardiac-seg) (cardiac MRI → ejection fraction, honestly evaluated).

## The honest question
The easy demo trains on real defects and reports a flattering number. The real questions: can a model
trained on **synthetic** defects detect **real** ones, *how much does it lose*, and is the detector
**calibrated** under that shift? The contribution is the evaluation rigor the field lacks (no
standardized sim-to-real 3D-anomaly benchmark exists), not a leaderboard score.

## Approach
- **Task:** unsupervised anomaly detection — train on "good" only, detect + localize defects.
- **Model:** a reconstruction autoencoder (VAE) that rebuilds normal surfaces; reconstruction error +
  latent distance = the anomaly score. Built (ported from acoustic anomaly work), not borrowed.
- **Data representation:** the scan's organized position map (per-pixel x,y,z) kept as an image grid,
  so per-pixel reconstruction error gives pixel-level localization directly.
- **Metrics:** image-level AUROC, pixel-level AU-PRO, calibration (ECE) under shift, per-condition
  diagnostics. Later: the sim-to-real triad (real ceiling / synth-only / synth + domain-adaptation).

## Status
🚧 **Stage 0, private.** Standing up the anomaly spine + evaluation harness on **real** data
(MVTec 3D-AD) before any synthetic generation. "Look at the data" step done (see `scripts/`). Goes
public when the spine is presentable. Full staged plan → [`docs/PLAN.md`](docs/PLAN.md).

## Data
**MVTec 3D-AD** (Bergmann et al., VISAPP 2022) — real industrial 3D anomaly benchmark, 10 object
categories scanned with a Zivid structured-light sensor (per-pixel rgb + xyz position map + defect
masks). The data lives **outside the repo** (licensing + size): set one root in `paths.yaml`, drop the
download under `<root>/raw/mvtec_3d_anomaly_detection/`. CC BY-NC-SA 4.0 — non-commercial, attribution
required; not redistributed here.

## Layout
```
mirage/
  scripts/     standalone tools (look_at_sample.py — view a scan as a 3D point cloud)
  docs/        PLAN.md — the staged build plan
  pyproject.toml / requirements.txt   packaging + deps
```
Pipeline / model / viewer directories are added only when they hold real code — no speculative folders.

## Quickstart
```bash
conda create -n mirage python=3.11 -y && conda activate mirage
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build (Blackwell/RTX 5090)
pip install -e .
# look at one scan (needs the dataset under <root>/raw/mvtec_3d_anomaly_detection/):
python scripts/look_at_sample.py --cat bagel --split test --defect hole --idx 0 --mask
```

## How it's built
Agent-driven build, human-owned judgment — the modeling, measurement correctness, and evaluation are
mine; coding agents scaffold the plumbing. Data-structure reasoning and evaluation discipline carry
over from prior ML work; the 3D specifics I learn as I go.

## License
Code: see [LICENSE](LICENSE). The dataset is **not** included and carries its own license
(CC BY-NC-SA 4.0) — obtain it from [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).
