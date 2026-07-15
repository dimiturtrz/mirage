# Deploy cost-model — the detector choice is set by the accelerator, and the numbers say so

The engine-first stance always deferred a "does it deploy?" answer to when hardware lands. This closes
the analytical half of that gap: measure each detector's real compute/memory footprint now, project it
onto candidate edge accelerators from their datasheets, and read the deploy-driven detector choice off
numbers instead of prose. **Integrity boundary up front: this is a projection, not an on-device
measurement.** Latency is `FLOPs / peak-TOPS` scaled by a stated 30–70% efficiency band — a range,
never a measured FPS — and it is reported only where a compute rate is actually in the source. What is
measured (params, FLOPs, activation, bank bytes) is labelled measured; what is projected is labelled
projected. No on-device benchmark is claimed.

## What was measured (`surfscan.deploy profile`, torch FlopCounterMode, 256²)

| detector | params (M) | GFLOPs | peak-act (MB) | int8 disk (MB) | bank |
|---|---|---|---|---|---|
| ConvVAE (xyz recon) | 30.8 | 2.3 | 15.7 | 30.8 | — |
| DRAEM (rgb) | 15.5 | 48.4 | 76.9 | 15.5 | — |
| feature-recon (backbone→L2 + AE) | 6.2 | 13.9 | 19.9 | 6.2 | — |
| PatchCore (backbone→L3) | 24.9 | 24.0 | 23.6 | 24.9 | 20k×1536 |

Backbones are counted **truncated to the layer the detector reads** (wide_resnet50_2 → layer2 for
feat-recon, → layer3 for PatchCore); layer4 + fc never run at inference and are pruned from the
footprint. The headline is that **params and FLOPs do not track each other** — ConvVAE is param-heavy
but compute-light (30.8M / 2.3G), DRAEM the opposite (15.5M / 48.4G) — so no single "model size" number
ranks deployability. You need both, plus the activation peak that sets the working-set floor.

## The bank is the load-bearing constraint, not the FLOPs

PatchCore's forward is unremarkable (24.9M / 24G). Its deploy cost is the **stored feature bank**:
greedy coreset at 0.1 of a 200k-patch pool → **exactly 20 000 vectors × 1536-d** (measured: bagel /
foam / tire all hit the pool cap and fit to (20000, 1536)). That bank is:

| representation | size | fits ~8 MiB SRAM? |
|---|---|---|
| fp32 (as fit) | **117 MiB** | no (14×over) |
| int8 | 29 MiB | no |
| int8 + PQ (m=8, b=8) | **1.65 MiB** | yes — at a cost |

The PQ byte formula reproduces the PatchCore-Lite literature anchor exactly (K=10k, d=256 → 334.1 KiB /
9.77 MiB fp32 [S3]), which is the cross-check that the model is arithmetic, not hand-waving. PQ makes
the bank *fit*, but [S3] measures the accuracy it costs: image-AUROC **0.95 → 0.86 (~9 pp)** while
localization survives (P-ROC 0.96 → 0.94). And even quantized, the **search** doesn't fully map: the
distance field reduces to a frozen Conv1×1 (W=−2B, bias=‖b‖²) that runs on-accelerator, but the
argmin/top-k over the bank dimension is a **host-side residue on every commodity NPU** (Coral/Hailo: not
in the op graph; RKNN: ArgMin yes, TopK no [S1][S4-land]).

## The projection: two orthogonal verdicts (`surfscan.deploy project`)

The join of measured cost × [cited accelerator specs](../../research/deep_dives/2026-07-08_edge_accelerator_landscape.md)
splits into two axes that must not be conflated:

**Op-fit — does the compute graph run on-accelerator?**
- **Bank-free conv (ConvVAE, feat-recon, DRAEM): fits-ondevice on all four.** Pure conv, fully
  int8-quantizable, no data-dependent lookup → maps onto Coral / Hailo / RKNN / Jetson alike.
- **PatchCore / fused: backbone + distance-matmul on-accel, argmin/top-k on HOST** on every commodity
  NPU. Only **Jetson** (native TopK/ArgMin, GB-scale LPDDR) runs the lookup end-to-end on-device.

**On-chip fit — does the working set fit, kept separate from op-fit.** Exceeding Coral's 8 MiB SRAM
means off-chip host streaming (a latency penalty), *not* undeployable; a DRAM-free Hailo would instead
be a hard ceiling, but its on-die capacity is unsourced. Against Coral's known 8 MiB, **every detector's
working set streams off-chip** — the smallest (feat-recon, 24.9 MiB) is already 311% of the envelope,
DRAEM is 1102%. The 8 MiB class is simply too small for any of these detectors' int8 weights +
activations to stay resident.

## The detector-choice thesis, in numbers

The accuracy table alone says PatchCore (0.902 AU-PRO) and feat-recon (0.907) are near-ties. The cost
model says they are **not** the same deploy decision:

- **feature-recon is the cheap, runs-entirely-on-NPU path** — 6.2M params (¼ of PatchCore), no bank, no
  host tail, quantizes onto a Coral/Hailo — at *equal* localization (0.907 vs 0.902).
- **PatchCore buys robustness, and pays for it in deployability** — its bank blows the on-chip envelope
  and its kNN tail is host-side, so it is fully-on-device only on a Jetson-class GPU SoC.

The tension is the field's known one, and the measured edge numbers corroborate it: bank-free
conv/distillation models (EfficientAD-class) are ~3–4× faster and fully-NPU-mappable, but their
detection can **collapse under distribution shift** — arxiv 2605.24251 measures EfficientAD
**0.925 → 0.529** in a continual setting where PatchCore holds 0.925 [S4]. Our own sim-to-real triad is
the same warning from our data: reconstruction/feature methods carry a real
[sim-to-real gap](../sim-to-real/2026-07-08_sim-to-real-triad.md) that a memory-bank comparison-to-normal
resists. Notably, **no published work deploys the bank lookup on a fixed-function NPU** [S4] — every
measured edge PatchCore ran the NN on host CPU or a Jetson GPU.

So the deploy-driven choice is not "pick the top AU-PRO"; it is:

> **Fixed-function NPU (Coral / Hailo)** → a bank-free conv detector (feature-recon), accepting the
> shift-robustness risk, because the memory bank cannot live or be searched there.
> **GPU SoC (Jetson)** → PatchCore / fused runs whole and its +2.7 pp AU-PRO
> [weighted-fusion](../detector/2026-07-15_weighted-fusion.md) win is bankable on-device.

That is the PLAN's deploy-driven-detector thesis reproduced with measured footprints and cited
datasheets rather than assertion — which was the point of the exercise.

## Honest limits

- **Projection, not measurement.** No latency was measured on any accelerator. The band assumes a
  30–70% sustained-efficiency fraction of peak TOPS; real kernels may fall outside it.
- **Unsourced specs stay `None`.** Coral / RKNN int8 TOPS and the Hailo on-die SRAM capacity sit behind
  vendor portals and were not fetched, so those pairs carry a memory/op-fit verdict but no latency band.
- **The robustness numbers (0.925 / 0.529) are the field's, not ours** [S4] — external corroboration of
  the bank-free-vs-bank tradeoff, not a measurement on MVTec-3D through our harness.
- **Roofline class is omitted**: it needs memory bandwidth, which the sources don't quote for most of
  these parts.

Structured sources: The single structured source `docs/deployment.json` carries the per-model `models`, the `bank`,
the cited `substrates`, and the 20-pair `matrix`. Research: [edge-accelerator
landscape](../../research/deep_dives/2026-07-08_edge_accelerator_landscape.md) [S1–S8],
[PatchCore-bank-on-NPU](../../research/deep_dives/2026-07-08_patchcore_bank_on_npu.md) [S3][S4].
