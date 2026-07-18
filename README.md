# mirage — 3D surface-defect anomaly detection with a measured sim-to-real gap

**Personal learning project.** Anomaly detection on 3D surface scans (MVTec 3D-AD), training only on
*good* examples — plus a synthetic-defect generator and a measured sim-to-real gap. Edge-deployable.

> Working name `mirage` (synthetic that must survive contact with the real). Repo folder may differ.

![mirage anomaly viewer — the working detector's heatmap on a held-out bagel](docs/media/viewer-anomaly.png)

*The working detector (PatchCore feature memory bank) scoring a held-out bagel — bright = anomalous,
lit at the defect. Rotatable in-browser → [`pointcloud-viewer/`](pointcloud-viewer/). (Scan from
[MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad), CC BY-NC-SA.)*

mirage detects **defects on 3D surface scans**, training only on *good* examples. Stage 0 stands up the
detector + an eval harness on **real** data and runs a like-for-like comparison of methods; Stage 1 adds
a **synthetic-defect generator** and measures the **sim-to-real gap**.

It's also how **I'm** ramping — into 3D perception first, and lately a second axis. The whole point is the
sim-to-real *eval*, not any one detector, so I keep testing that it isn't perception-specific: the same
discipline now points at **control** too — a policy learned in sim, the same gap measured when the dynamics
shift under it (a deliberate ramp, [further down](#the-same-eval-on-a-second-axis--control-a-ramp)). All of
it on public data, the data-engine + evaluation discipline carried from prior acoustic-detection work, the
new modalities learned as I go. Full plan → **[docs/PLAN.md](docs/PLAN.md)**.

## Stage 0 result — the measured investigation
Not a leaderboard number — the *measured contrast*. Per-category models, scored through one verified
eval harness (image-AUROC + pixel-AU-PRO). Full table + diagnostics → [`docs/RESULTS.md`](docs/RESULTS.md).

| method | paradigm | pixel AU-PRO |
|---|---|---|
| VAE / inpaint / DRAEM / fused (4+ variants) | reconstruct pixels → error | **0.095** (≈ random) |
| **PatchCore** (the deployable) | **memory bank** — nearest-neighbour to normal features, *no reconstruction* | **0.90** |
| feature-recon | reconstruct *features* → error | 0.91 |
| **fused** (our best) | PatchCore-rgb **+ 3D geometry** (BTF/FPFH), z-score fusion | **0.92** |
| SOTA (multimodal fusion, *papers*) | RGB + 3D geometry fusion (M3DM 0.96 → DCRDF-Net 0.99) | ~0.96–0.99 |

**The finding:** pixel-reconstruction is ≈ random at localization — *measured* across four variants,
then *diagnosed*: raw-residual tracks geometric **complexity** (a curved rim is hard to rebuild), not
**defect-ness** (a smooth defect rebuilds fine → low residual). What works is **"compare to normal"** —
and two *different* paradigms get there: a **memory bank** (PatchCore: store normal features, score by
nearest-neighbour distance — no rebuilding at all) and **reconstruction moved into feature space**
(feature-recon). Both land ~0.90–0.91. Adding **3D-geometry fusion** lifts our best to **0.92** (+1.5pt over
rgb-only, matched, CIs near-disjoint) — the named SOTA lever, now *measured to pay* even with our
preprocessing-limited geometry; the rest of the gap is better geometry, not bank/resolution tuning. The
simplest *deployable* stays the rgb-only memory bank (no geometry pipeline) — what's worth showing is the
eval rigor + the measured mechanism, not a single number.

## Stage 1 — measuring the sim-to-real gap
The easy demo trains on real defects and reports a flattering number. The question that matters: can a model
trained on **synthetic** defects detect **real** ones, *how much does it lose*, and is it **calibrated**
under that shift? Stage 1 measures it with a **triad** — one segmenter, three label sources, one shared
real eval half, bootstrap CIs from a single deterministic run:

| arm | AU-PRO [95% CI] | ECE |
|---|---|---|
| real → real (ceiling) | 0.794 [0.772, 0.813] | 0.010 |
| synth → real (**the gap**) | 0.628 [0.604, 0.650] | 0.075 |
| synth + DA → real (AdaBN) | 0.643 [0.618, 0.669] | 0.076 |

**Sim-to-real gap = 0.166 AU-PRO [0.141, 0.190]** (paired bootstrap, clears 0 — a real gap). Domain
adaptation (AdaBN) adds only **+0.015 [−0.000, 0.030]** — not significant, and doesn't restore calibration:
a reported negative, kept. The lever that *did* move the number was **not overtraining on synthetic**
(early-stopping on a held-out synth val lifted synth→real 0.54 → 0.63). Full method + diagnosis →
[`docs/RESULTS.md`](docs/RESULTS.md).

Sim-to-real for 3D anomaly is only now being charted ([SiM3D](https://arxiv.org/abs/2506.21549), 2025 — the
first synthetic→real benchmark, single-instance CAD→real). What's still open — a physics-based generator
that shrinks the gap **at the source**, and a closed-loop curriculum — is what comes next.

## Stage 2 — a physics-based digital twin (a negative traced to its cause, then closed to parity)
Shrinking the gap *at the source* means a better generator. Stage 2 builds one: reconstruct a per-category
mean-surface mesh from the 244 real good scans, render it headless through **Isaac Sim / USD / Replicator**
(path-traced, back-projected to an organized xyz map, MVTec layout), and feed the *same* triad. Scored on the
controlled **shape-source** delta (twin_grid − synth: same on-the-fly grid defect, same real eval — only the
good-scan surface differs), on **geometry** (xyz, the twin's axis), it first came out **negative** (−0.192
AU-PRO): the reconstructed surface was a worse training substrate than the real scan. The rigorous spine then
traced *why* and removed it — two levers, each argued a priori (4-cat subset, single seed):

| shape-source (twin_grid − synth), xyz | value |
|---|---:|
| 256² baseline | −0.224 |
| + native resolution (512², jlc.1) | −0.067 |
| + Zivid sensor-capture model (jlc.2) | **+0.019** |

A higher-fidelity rebuild had already **not** moved it (Fork A, flat) — the deficit was never the mesh. It was
(1) detail **resampled away at 256²** — native resolution recovers +0.157, by removing classical synth's
real-surface advantage; and (2) the **render↔Zivid sensor domain gap** — a [physics-derived structured-light
model](core/data/dynamic/sensor_noise.py) (dropout at the measured ~11 % rate + grazing 1/cosθ noise, *not* a
tuned Gaussian) trains the twin on realistic sensor artifacts for a further +0.086. Together they lift the twin
from −0.224 to **parity** (+0.019). Honestly *parity, not dominance* — the twin now matches classical at the
source, it does not beat it; the result is the **closure and its mechanism** (single-seed/subset, foam a kept
outlier — directional pending full-10 + multi-seed). Full verdict →
[`interpretations/twin/2026-07-18_source-verdict.md`](interpretations/twin/2026-07-18_source-verdict.md).

## The same eval on a second axis — control (a ramp)
If the real contribution is the sim-to-real *eval* and not one detector, it shouldn't much care what the task
is. So I gave it a different one: learn a policy in simulation, then ask how much it loses when the world
stops matching the sim. Same shared `core`, a `(train, act)` policy contract sitting beside perception's
`(fit, score)`, one rollout spine — a policy behavior-cloned in nominal sim, rolled against a *real* whose
dynamics have drifted (a heavier payload, a weaker actuator). The gap is the [Stage 1](#stage-1--measuring-the-sim-to-real-gap)
idea again, now in task success.

It holds until it doesn't. Feedback soaks up a payload up to +30% at a **zero** gap — and zero spread across
seeds — then falls off a cliff: **60 ± 4.8 pp** at +50%, essentially gone by +60%. Not a slope, a threshold,
and it tips its hand first — the policy takes visibly longer to reach the goal while it's still succeeding,
before it starts missing outright. The full curve, method, and integrity live in [`control/`](control/README.md).

So I tried two levers. First the obvious one: train across *randomized* dynamics instead of one nominal sim.
It halves the gap at the cliff's onset (+40%: **17.8 → 7.7 pp**) but barely dents the deep collapse — because
this policy is dynamics-blind (it never sees a payload cue), randomization can buy robustness in the marginal
band but can't beat a hard actuator limit. Then the lever that actually follows from that diagnosis: give the
policy one step of proprioception (its last action and the velocity change it caused), let it identify the
plant online and compensate. That **erases the cliff up to +50% payload (60 → 0 pp)** and cuts the extreme
+60% case from 98 to 18 pp. The point isn't the toy number — it's the contrast: the fix for a dynamics shift
is to *observe* it, not to average over it in training. (It closes this cleanly because the toy plant is a
clean scalar gain; on a real robot that identification is much harder — the claim is directional.)

Then the test of whether any of this is real engineering: the `Env` seam. I dropped in a PhysX rigid body
(Isaac Sim — a real solver, same reach dynamics) and re-measured the whole curve *without changing a line*
of the policy, the spine, or the metric — only a new environment. The finding **transferred**: same
margin-then-cliff, and the headline +50% gap came back essentially unchanged (**60.0 pp** Euler → **62.5 pp**
PhysX). The one honest difference — the cliff sits a touch earlier under real damping (+40%: 17.8 → 32.5 pp)
— is the expected direction, more faithful physics being less forgiving, not a contradiction.

Honest read: still simplified reach dynamics on both rungs, not a robot arm with contact and sensing — so
it's "the mechanism holds across two solvers", not a robotics result. But the seam is no longer a promise;
it's measured.

## Limits (measured, not assumed)
Edge-deployable and honestly benchmarked — **not** a production system. The gaps, measured rather than assumed:
- **Sim-to-real gap is real and open (0.166 AU-PRO).** Domain adaptation doesn't significantly close it
  (see [Stage 1](#stage-1--measuring-the-sim-to-real-gap)), and the [Stage 2](#stage-2--a-physics-based-digital-twin-the-honest-negative)
  digital twin doesn't beat classical synth at the source — and a higher-fidelity rebuild didn't close it
  (ablated), so the cap is the render-vs-sensor domain gap at 256², not mesh resolution. Native-res eval +
  render-domain adaptation are the open levers.
- **The deployable is rgb-only PatchCore (greedy coreset, 0.90).** Adding **3D-geometry fusion** is measured
  at **+1.5pt → 0.92** (our best) even with preprocessing-limited BTF; the remaining gap to SOTA (~0.96) is
  *better* geometry (native-res FPFH / learned point features), not bank or resolution tuning (rgb tops ~0.93).
  Fused needs the geometry pipeline (xyz + FPFH banks), so rgb-only stays the simplest edge path.
- **Our BTF underperforms paper-BTF** (0.65 vs ~0.96): FPFH runs on the 256-resized / grazing-noisy processed
  cloud, not native-resolution clean point clouds. A preprocessing limit, diagnosed, not a method failure.
- **3D is a ramp.** The data-engine + eval discipline carry from prior work; the 3D-perception specifics are
  the genuinely new skill, learned as I go.
- **The control axis is a ramp.** The sim-to-real policy gap (60 ± 4.8 pp @ +50% payload,
  [above](#the-same-eval-on-a-second-axis--control-a-ramp)) is measured on a numpy point-mass and re-measured
  on a PhysX rigid body (Isaac Sim) via the `Env` seam — the headline transfers (60.0 → 62.5 pp). Domain
  randomization only nudges it (dynamics-blind policy); an **adaptive policy that senses the payload closes
  the cliff to +50% (60 → 0 pp)**, +60% residual 18 pp — the honest contrast (observe the shift, don't
  average over it). Both rungs are simplified reach dynamics, and the toy plant makes online system-ID easy,
  so the adaptive claim is directional, not "sim-to-real solved" — the honest ceiling.
- **Not a device.** Public research data only; edge deploy is real (ONNX), production-scale serving is not claimed.

## Data
**MVTec 3D-AD** (Bergmann et al., VISAPP 2022) — real industrial 3D anomaly benchmark, 10 categories
scanned with a Zivid structured-light sensor (per-pixel rgb + xyz position map + defect masks). Lives
**outside the repo** (licensing + size): set one root in `paths.yaml`, drop the download under
`<root>/raw/mvtec_3d_anomaly_detection/`. CC BY-NC-SA 4.0 — non-commercial, attribution required; not
redistributed here.

## Layout
`core/` (reusable engine) + `surfscan/` (perception science) + `control/` (control leg). The synthetic-data
engine lives **by role**, not in a runtime-named bucket — the Isaac twin generator under `core/data/dynamic/`,
the PhysX control env under `control/sim/`; both behind the opt-in `sim` extra (only the *dependency* is "sim"):
```
core/                  the reusable, method-agnostic engine
  config.py            data root from paths.yaml (-> raw/ + processed/ + synth/)
  data/                the data layer — real ingest + synthetic generation, one store
    static/            adapters (mvtec · synth) + preprocess + unified store + GPU-resident dataset
    dynamic/           generation — classical Defects + the twin pipeline (reconstruct mesh → twin_geom lib → Isaac render)
  method.py            the (fit, score) contract the perception harness scores
  policy.py rollout.py the (train, act) contract + rollout spine the control leg scores
surfscan/              the perception science layer
  models/              vae · inpaint · draem · feat_recon · patchcore · fpfh_bank (BTF)
  training/            train spine + hparams + losses
  experiments/         run_* — per-method benchmark runners (produce the AU-PRO board)
  evaluation/          harness (spine) · metrics · scoring · diagnostics · sync_numbers
  deploy/              profile (footprints) · accelerators (typed loader) · bank · fit (verdict matrix)
  visualization/       show.py (3D viewer) + export_web.py (web-viewer data)
control/               the control leg — point_mass env · PD/adaptive experts · bc · act · diffusion policies · policy-gap experiments
  sim/                 the PhysX Isaac reach env (isaac_reach) + gap runner (isaac_gap) — fidelity rung
deploy/            browsable fit explorer (LIVE → dimiturtrz.github.io/mirage) — models_params · accelerators/<type> · fit_matrix + viewer
docs/                  PLAN.md · RESULTS.md (+ RESULTS.json canonical) · STRUCTURE.md
learning/ research/ tests/ (unit + integration) · pointcloud-viewer/
```

## Quickstart
```bash
uv sync --extra features          # .venv + deps; torch/torchvision cu130 (Blackwell/RTX 5090) auto-resolved
cp paths.example.yaml paths.yaml          # set `data:` to your data root
uv run python -m core.data.static.store        # consolidate raw -> processed (needs the dataset)

# the working detector, scored through the eval harness (logs params/metrics to MLflow):
uv run python -m surfscan.run patchcore   # image-AUROC + AU-PRO + per-defect, all 10 categories
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare every method run in the browser

# see it: the defect glows under the working detector (vs the VAE's backwards residual):
uv run python -m surfscan.viz show --cat bagel --split test --defect hole --idx 0 --processed --patchcore
```
Experiment tracking is **MLflow** (canonical store: `mlflow.db` + `mlartifacts/`, gitignored) — every
method/training run logs params, metrics (per-category + per-defect), the aggregate, and trained models.

## Tests
```bash
uv run pytest tests/ -q        # unit (equivalence-class) + integration (module pairs)
```
Two layers: a wide **unit** base (partition each input space, one representative per class + boundaries) and
an **integration** layer over module pairs (A's output is a valid B input). `tests/unit/` **mirrors the
source tree** — enforced by the arch-fitness gate — so a module's tests live where the module does.

## Quality gates
The static-analysis gates — style/bugs, dead code, import layering, architecture fitness, module shape,
duplication — run in one command locally and identically in CI:
```bash
uvx nox -s lint        # ruff · vulture · import-linter · graph --assert · ast-grep · jscpd
```
(tests are the section above; CI runs both.) The gate **analyzers** install as a version-pinned package
(`sdlc-devtools`), and their config + the nox/CI/pre-commit runners come from an in-house copier template
([sdlc-scaffold](https://github.com/dimiturtrz/sdlc-scaffold)) — refresh with `uvx copier update` (both
pinned in `.copier-answers.yml`). Template-owned: fix a gate upstream, don't hand-edit it to pass.

### Local hooks
The same gates CI enforces bind to git events via pre-commit. Install **both** stages:
```bash
pre-commit install                        # commit stage — fast static gates (ruff/vulture/arch-fitness/…)
pre-commit install --hook-type pre-push   # push stage — fast unit suite (tests/unit)
```
The pre-push hook runs `pytest tests/unit`, so a change that lints clean but breaks a test **contract** (a
signature change a mirror test still calls the old way) is caught before the push, not after CI goes red —
push-only (not every commit), unit-only (integration/e2e + the coverage floor stay CI jobs). The default
`pre-commit install` does **not** wire pre-push; run the second line once per clone.

## How it's built
Agent-driven build, human-owned judgment — the modeling, measurement correctness, and evaluation are
mine; coding agents scaffold the plumbing. Data-structure reasoning and evaluation discipline carry over
from prior ML work; the 3D specifics I learn as I go ([`learning/`](learning/)).

## Architecture
Interactive import graph of `core` + `surfscan` + `control` — **live** at
[dimiturtrz.github.io/mirage/architecture/](https://dimiturtrz.github.io/mirage/architecture/), composed into
the `deploy-fit-explorer` Pages workflow beside the fit-explorer views. `nox -s archmap` regenerates
`docs/architecture/graph.json` (committed + diffable — the architecture-erosion record; the `index.html`
viewer is gitignored, rebuilt on demand).

## References
- **MVTec 3D-AD** — Bergmann et al., *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and
  Localization*, VISAPP 2022.
- **PatchCore** — Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.
- **DRAEM** — Zavrtanik et al., *DRÆM — A Discriminatively Trained Reconstruction Embedding for Surface
  Anomaly Detection*, ICCV 2021.
- **Reverse Distillation (RD4AD)** — Deng & Li, *Anomaly Detection via Reverse Distillation from One-Class
  Embedding*, CVPR 2022. (Basis for feature-recon.)
- **BTF** — Horwitz & Hoshen, *Back to the Feature: Classical 3D Features are (Almost) All You Need for 3D
  Anomaly Detection*, CVPR-W (VAND) 2023.
- **M3DM** — Wang et al., *Multimodal Industrial Anomaly Detection via Hybrid Fusion*, CVPR 2023.
- **DCRDF-Net** (current SOTA) — Wang, Chen & Zhang, *A Dual-Channel Reverse-Distillation Fusion Network for
  3D Industrial Anomaly Detection*, Sensors 26(2):412, 2026 ([doi:10.3390/s26020412](https://doi.org/10.3390/s26020412)).
- **SiM3D** — the first synthetic→real 3D-anomaly benchmark (single-instance, CAD→real),
  [arXiv:2506.21549](https://arxiv.org/abs/2506.21549), 2025.

## License
Code: see [LICENSE](LICENSE). The dataset is **not** included and carries its own license
(CC BY-NC-SA 4.0) — obtain it from [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).
