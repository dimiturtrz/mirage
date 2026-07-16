# deploy — fit explorer

**Live — two views on one Pages site** (`.github/workflows/pages.yml` assembles both branches):
- https://dimiturtrz.github.io/mirage/ — **stable** (`main`)
- https://dimiturtrz.github.io/mirage/preview/ — **latest** (`dev`)

A push to either branch rebuilds the whole site from both branches' current `deploy/`; the header carries a
link between the two. One-time setup: enable Pages at **Settings → Pages → Build and deployment → Source:
"GitHub Actions"** (the default Actions token can't create the site itself); after that it deploys
automatically. The data is license-clean — our own measured
footprints + cited **public** vendor datasheets, zero MVTec-derived content — so unlike `pointcloud-viewer/`
it is safe to host publicly.

Which anomaly detector ships on which edge accelerator, and **why**. This is the deploy analysis made
browsable: measured model footprints crossed against typed, cited accelerator specs into a prescriptive
`(detector × accelerator × options) → verdict + reason` matrix, rendered by a self-contained viewer.

The load-bearing claim is architectural, not a FLOP count: a **conv** graph quantizes and ships to any
NPU; a PatchCore **bank kNN** tail (argmin/top-k over the bank) is a **host residue** on every commodity
NPU and runs whole only on a GPU SoC; **attention** is not in a fixed-function op-set at all.

That claim is **compile-verified**, not just cited: a representative graph per op-class is pushed through the
real **RKNN** (`rknn-toolkit2`) and **Coral** (`edgetpu_compiler`) toolchains — static partitioners, no
silicon — and both confirm conv maps to the accelerator while the bank's ArgMin/TopK fall to the host
(`compile_ops.json`; deep-dive `2026-07-16_edge_op_support_compiled.md`). Cards for those two accelerators
carry a **⚙ compiled** badge. Hailo's compiler is Developer-Zone-gated, so Hailo op-support stays cited; and
one nuance surfaced — a bare attention block *fuses* to the RK3588 NPU, so "unsupported" is a detector-level
verdict (Point-MAE's kNN tokenizer stays host), not a claim that no attention op exists.

## Layout
```
deploy/
  models_params.json              generated — measured components + detector op-classes + bank model
  accelerators/
    cpu_params.json               hand-authored, cited — one file per accelerator TYPE, instances inside
    gpu_params.json               (rpi5 · jetson-orin · coral/hailo · rknn)
    npu_fixed_params.json
    npu_soc_params.json
  fit_matrix.json                 generated — the (detector × accelerator × options) verdict grid
  compile_ops.json                generated — op-support compile-verified on real RKNN + Coral toolchains
  index.html                      the viewer (three-free, no build step)
```

The **type** file carries the op-support contract (how it runs each op-class: native / host-tail /
unsupported); the **instance** carries its datasheet numbers and memory model (which varies within a
type — Coral streams from a scratchpad, Hailo is on-die-only, both `npu_fixed`). Numbers behind a vendor
portal are `null`, never guessed; every instance cites its source `[S#]` in the
[edge-accelerator landscape](../research/deep_dives/2026-07-08_edge_accelerator_landscape.md).

## Regenerate
```bash
python -m surfscan.deploy profile     # measure footprints -> models_params.json
python -m surfscan.deploy fit         # cross with typed accelerators -> fit_matrix.json
python -m surfscan.deploy compile_probe --build /tmp/g   # emit representative graphs; then, on a Linux box
#   with the vendor SDKs, compile them and: compile_probe --rknn-log R --edgetpu-log E -> compile_ops.json
```
The accelerator `*_params.json` are source (hand-authored); `profile`/`fit` never touch them. `profile`
needs a GPU for the activation peak and `external/M3DM` for the pointmae transformer detector — both
degrade gracefully (act = NaN off-CUDA; pointmae skipped if M3DM absent). `fit` is a pure CPU join and a
test asserts the committed `fit_matrix.json` matches a fresh recompute, so it can't drift.

## View
Live at the URL above, or locally:
```bash
cd deploy && python -m http.server 8000   # fetch needs http, not file://
# open http://localhost:8000
```
Pick an accelerator, toggle **quantization** (int8/fp32) and the **export toolchain** (ONNX/eager); the
per-detector verdicts and the full matrix re-render. The viewer is a pure renderer — all fit logic lives
in `surfscan/deploy/fit.py` (Python is the single source of truth), so the page can't drift from the model.

## Integrity
Latency is a **projection** — FLOPs / (peak int8-TOPS × a 30–70% efficiency band) — reported only where a
TOPS number is cited, and it is **never a measured FPS**. The op-fit and memory-fit verdicts are exact and
hold with no TOPS number. On-device measurement is a separate, hardware-gated step (bead `0sq`).
