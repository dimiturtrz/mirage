# synthscape

**Synthetic-driven 3D point-cloud perception with sim-to-real evaluation and edge deployment.**

Train perception on *synthetic* LiDAR scenes I generate, prove it on *real* data it never saw
(KITTI / nuScenes), measure the domain gap honestly, and ship it to run in real time on edge.

> Working name — rename freely.

## Thesis

The hard part of perception isn't fitting the training set — it's holding up on real, unseen
conditions. Real 3D labels are scarce and expensive; synthetic scenes are cheap and perfectly
labelled. So the whole pipeline serves one question: **does a model trained on synthetic point
clouds generalize to the real world — and can I prove it and deploy it?**

That makes three things first-class, not afterthoughts:
1. **Synthetic data generation** (the cheap, perfectly-labelled source).
2. **Sim-to-real evaluation** (the real yardstick — domain-gap diagnostics, calibration, not a
   flattering in-distribution number).
3. **Edge deployment** (ONNX/TensorRT, latency/FPS benchmark — perception that actually runs).

## Status

🚧 Stage 1 in progress (private). See [`docs/PLAN.md`](docs/PLAN.md) for the staged roadmap.

## Layout

```
synthscape/
  sim/         synthetic LiDAR scene generation (+ domain randomization)
  data/        real-dataset loaders (KITTI / nuScenes) + format converters
  models/      3D point-cloud models (detection / semantic segmentation)
  training/    training loops, curricula
  evaluation/  sim-to-real diagnostics, calibration, metrics
  deploy/      ONNX/TensorRT export, latency/FPS benchmark, containerized inference
configs/       experiment configs
tests/         unit / integration
docs/          PLAN.md (staged build plan)
```

## Quickstart

```bash
pip install -e .
# (stubs only for now — see docs/PLAN.md)
```
