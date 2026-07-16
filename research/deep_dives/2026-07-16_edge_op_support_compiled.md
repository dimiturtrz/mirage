# Compile-verified edge op-support — conv native, kNN tail is a host residue

**Date:** 2026-07-16 · **Bead:** j5s.5 · **Artifact:** `deploy/compile_ops.json` · **Generator:** `surfscan/deploy/compile_probe.py`

## Question

The deploy fit matrix asserts a per-op-class support verdict for each accelerator:
`conv_native → native`, `bank_lookup → host_tail` (Conv1x1 distance on-accelerator, argmin/top-k on the host),
`transformer_attention → unsupported`. Until now that assertion was **cited** — read off vendor op-support
tables in the [edge-accelerator landscape](2026-07-08_edge_accelerator_landscape.md). This step **compiles**
it: does a real vendor toolchain actually partition the graph the way we claim?

## Method — representative graph per op-class, no silicon

Both compilers are **static graph partitioners** — they assign each op to the accelerator or the host at
build time, no board attached. So op-support is verifiable on a Linux dev box; only on-device *latency*
needs hardware (bead `0sq`).

- **RKNN** — `rknn-toolkit2` 2.3.2, target `rk3588`, simulator. The build log tags each op `NPU` or `CPU`.
- **Coral** — `edgetpu_compiler` 16.0.384591198 (extracted rootless from the Coral apt `.deb`; no Edge TPU
  attached). The op log tags each op `Mapped to Edge TPU` or gives a CPU-fallback reason.
- **Hailo** — dataflow compiler is Developer-Zone-gated (login + license); **stays cited**.

Three representative ONNX graphs carry the **exact op types** `export_ops.json` recorded per class, so the
compiler's verdict on the representative transfers to the real detectors:
- `conv_rep` — Conv / BatchNorm / Relu / MaxPool / Add (a resnet block).
- `bank_tail` — the PatchCore edge tail (bead 7cx): distance = frozen **Conv1x1** (`W=-2B`, `bias=‖b‖²`),
  selection = **ArgMin + TopK**.
- `attn_rep` — MatMul / Softmax / MatMul scaled dot-product attention.

## Result

| representative (op-class) | RKNN rk3588 | Coral Edge TPU |
|---|---|---|
| `conv_rep` (conv_native) | ConvRelu, MaxPool, ConvRelu → **all NPU** | CONV_2D×2, MAX_POOL_2D, ADD, TRANSPOSE → **all Edge TPU** |
| `bank_tail` distance Conv1x1 | → **NPU** | CONV_2D → **Edge TPU** |
| `bank_tail` selection ArgMin/TopK | → **CPU** | TOPK_V2 *"not supported"*, ARG_MIN *"unsupported data type"* → **CPU** |
| `attn_rep` (transformer) | fuses to **NPU** (`exSDPAttention`) | **unconvertible** — BatchMatMul won't int8-quantize to TFLite |

Coral's `bank_tail` summary: **1 op on Edge TPU, 6 on CPU** — the compiler amputates the graph at the first
unsupported op, exactly the "single-partition CPU fallback" the fit note describes.

## Reconciliation with the cited matrix

- **conv_native → native.** CONFIRMED on both toolchains. Every conv/pool/add op maps to the accelerator.
- **bank_lookup → host_tail.** CONFIRMED on both. The Conv1x1 distance maps; **ArgMin and TopK are rejected**
  and fall to the host. This is the load-bearing architectural claim of the whole deploy story, now compiled
  rather than cited.
- **transformer_attention → unsupported.** REFINED, not overturned. A **bare** SDPA block *fuses* to the
  RK3588 NPU (`exSDPAttention`), so "unsupported" is too strong for a lone attention op. But (a) Coral cannot
  int8-quantize attention to TFLite at all, and (b) the real Point-MAE detector carries FPS/kNN tokenizer ops
  (TopK / ScatterND / ArgMax, per `export_ops.json`) that stay on the host — so the **detector-level** verdict
  (not accelerator-native end to end) holds, for a subtler reason than "no attention op exists." The matrix
  label is unchanged; the *reason* is now precise.

## Integrity

This verifies **op-support** (compile-time partition), not throughput. Latency stays a projection until a
measured on-device anchor (bead `0sq`). Hailo op-support stays cited (SDK-gated). Two of the three
fixed-function targets (Coral, RKNN) are now compile-verified.

## Reproduce

```bash
python -m surfscan.deploy compile_probe --build /tmp/g          # emit conv_rep/bank_tail/attn_rep .onnx
# on a Linux box with the SDKs (rknn-toolkit2; edgetpu_compiler):
#   rknn: load_onnx + build(do_quantization=False) per graph, capture the build log
#   coral: onnx2tf -oiqt (supply a calibration .npy via -cind) then edgetpu_compiler -s
python -m surfscan.deploy compile_probe --rknn-log R.log --edgetpu-log E.log   # -> deploy/compile_ops.json
```
