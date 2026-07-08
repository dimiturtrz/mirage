# Edge AI Accelerator Landscape — op-support constraints for anomaly-detection deploy

**Date**: 2026-07-08
**Status**: settled
**Supersedes**: (none — no prior edge-accelerator deep-dive)

## TL;DR

Coral EdgeTPU / Hailo-8 / generic NPUs (RKNN) are **dataflow/quantized-conv accelerators**: full int8, fixed graph, no data-dependent gather/distance/top-k on-device. A PatchCore kNN-vs-stored-bank lookup is **inherently host-CPU** on all three (bank streaming + argmin/top-k not accelerated), and PatchCore's fp32 memory bank breaks the int8-everywhere rule. A pure-conv int8 autoencoder maps cleanly onto all of them. Only Jetson/TensorRT is general enough (full ONNX, fp16/int8, gather/top-k native) to run both — but it's a GPU SoC, not a commodity NPU.

## Question

For each commodity edge accelerator (Coral EdgeTPU, Hailo-8/8L, generic NPU e.g. RKNN) vs Jetson/TensorRT: which ops run on-accelerator vs forced to host CPU? Specifically for a PatchCore detector (frozen conv backbone + kNN against a stored bank of ~10⁴–10⁵ feature vectors) vs a pure-conv int8 autoencoder. Marshal facts; leave the deploy call to the reader.

## Findings

### 1. Google Coral EdgeTPU (USB Accelerator / Dev Board)

- **Full int8 quantization is mandatory.** "For compatibility with the Edge TPU, you must use either quantization-aware training (recommended) or full integer post-training quantization." No float compute on-device; float I/O tensors are allowed but the quantize/dequantize ops run on **CPU** [S1].
- **Single-partition CPU fallback.** "the Edge TPU compiler cannot partition the model more than once, so as soon as an unsupported operation occurs, that operation and everything after it executes on the CPU, even if supported operations occur later." [S1] — one unsupported op amputates the entire tail of the graph to host.
- **Supported-op table is small and conv-centric.** `FullyConnected` supported but "Output tensor is one-dimensional"; `Concatenation` supported only with restrictions (no constant inputs unless all-zeros) [S1].
- **Gather, ArgMin/ArgMax, TopK are NOT in the supported-ops table** → forced to host CPU [S1]. (Absence from Table 1 = unsupported, per the doc's own rule that only listed ops run on-TPU.)
- **Large matmul against a stored constant matrix**: no dedicated "matmul-vs-stored-bank" op. `FullyConnected` exists but a ~10⁴×D bank is a large weight tensor competing for the 8 MiB budget; and a kNN needs argmin/top-k over the result, which are host-only anyway.
- **On-chip memory ≈ 8 MiB SRAM** (scratchpad, compiler-assigned, not a HW cache) for weights+activations+instructions [S2][S5]. Larger models stream uncached params from host off-chip, with a large latency penalty [S2][S5]. A fp32 PatchCore bank of tens-of-thousands of D-dim vectors (e.g. 50k×272 fp32 ≈ 54 MB) vastly exceeds 8 MiB and isn't int8 → cannot live on-TPU.

### 2. Hailo-8 / Hailo-8L

- **TOPS**: Hailo-8 = **26 TOPS**; Hailo-8L = **up to 13 TOPS** [S6].
- **DRAM-free, all memory integrated on-die.** "Hailo-8 is the only AI accelerator... which... integrat[es] all required memory directly on the processor die." Hailo-8L is likewise "DRAM-free." [S6] — implies a hard on-chip capacity envelope; no large external bank streaming as a design point.
- **Compiler path**: ONNX → Dataflow Compiler (parse → optimize/quantize → compile) → HEF binary [S3][S7].
- **Quantization**: params "converted from float32 to int8" via calibration-set activation statistics [S3]. Int8 is the deployment representation.
- **No general per-op CPU fallback — the whole graph compiles to the accelerator, or you truncate.** The documented pattern for unsupported ops: "unsupported layers are ideally located at the first or last layers... specify the last CONV2D layers as output layers, and implement the missing layers in your application" — i.e. you cut the model at the last supported conv and **run pre/post-processing on the host CPU yourself** [S7]. There is no automatic mid-graph host offload like a runtime; it's a build-time truncation. `Reshape` between Conv and Dense is unsupported and must be handled via `end_node` [S3].
- **Consequence for a distance/top-k tail**: not expressible in the dataflow graph → must be host code after the conv backbone's HEF output.

### 3. Generic NPU class (Rockchip RKNN as representative)

- **RKNN op-support (Toolkit2 v1.6.0)**: `Gather` supported, `ArgMax` supported, `ArgMin` supported, but **`TopK` explicitly "Not Supported"** (CPU/unsupported) [S4]. Nearest-neighbor / `cdist` / pairwise-distance ops are **not listed at all** [S4].
- **Generalization across the NPU class**: a kNN / cdist / large stored-feature-bank lookup is **not a native accelerator primitive** on this class. These NPUs accelerate conv/matmul/elementwise/pooling; the data-dependent, gather-and-sort-heavy distance search is a host-CPU operation. Even where `Gather`/`ArgMax` are individually listed, a full kNN needs top-k (unsupported) and a fp32 bank, so the lookup as a whole falls to host. (Stated as class-level inference from RKNN's list + the shared dataflow-NPU architecture; not separately verified on every vendor's NPU.)

### 4. Jetson / TensorRT (contrast)

- **General-purpose GPU inference, not a fixed-function NPU.** TensorRT imports full ONNX and supports mixed precision **FP32/FP16/BF16/FP8/INT8/FP4/INT4** [S8] — int8 is optional, not mandatory.
- **Data-dependent ops are native**: `TopK` is a first-class TensorRT layer (limit K ≤ 3840) [S8]; `Gather`/`ArgMax` are in the ONNX-TensorRT operator matrix; anything unsupported can be a custom **plugin** running on the same GPU [S8].
- **No 8-MiB-class on-chip envelope**: Jetson modules carry GB-scale LPDDR shared between CPU and GPU, so a tens-of-thousands-vector feature bank fits in device memory and a distance/argmin kernel runs on-GPU.
- **Net**: the op-restriction/quantization constraints that bind Coral/Hailo/RKNN largely **do not apply** to Jetson/TensorRT. It is the outlier that can run a PatchCore-style lookup end-to-end on-device — at the cost of being a larger, pricier GPU SoC rather than a commodity USB/M.2 NPU.

### Marshalled comparison for the reader's decision

| Component | Coral EdgeTPU | Hailo-8/8L | Generic NPU (RKNN) | Jetson/TensorRT |
|---|---|---|---|---|
| Frozen conv backbone (int8) | ✅ on-TPU | ✅ on-accel | ✅ on-NPU | ✅ on-GPU |
| kNN vs stored feature bank | ❌ host CPU | ❌ host CPU | ❌ host CPU | ✅ on-GPU (feasible) |
| — top-k / argmin over distances | ❌ not in op table | ❌ not in dataflow graph | ⚠️ TopK unsupported (ArgMin ok) | ✅ native TopK/ArgMin |
| — fp32 bank storage | ❌ >8 MiB, not int8 | ❌ DRAM-free envelope | ❌ not int8 / not native | ✅ fits in LPDDR |
| Pure-conv int8 autoencoder | ✅ clean | ✅ clean | ✅ clean | ✅ clean |

Where the facts clearly point (reader decides): PatchCore's **backbone** accelerates everywhere; its **kNN-vs-bank tail is host-CPU on the three commodity NPUs** and only native on Jetson. A pure-conv int8 autoencoder (no lookup, fully quantizable) is the one design that maps cleanly onto *all four* — including the cheap commodity NPUs.

## Open questions

- Exact Hailo-8 / 8L on-chip SRAM capacity in MiB (datasheets S9/S10 cited but not fetched — behind Hailo file portal; the "all-memory-on-die / DRAM-free" claim is confirmed but the number isn't quoted here).
- Coral's absolute model-size ceiling beyond the 8 MiB cache: off-chip streaming works but the practical PatchCore-bank latency penalty is unquantified here.
- Whether a *reduced* PatchCore (coreset subsampled to a small int8-quantized bank + approximate NN) could be coerced onto a commodity NPU — **now researched: see [2026-07-08_patchcore_bank_on_npu.md](2026-07-08_patchcore_bank_on_npu.md)**. Short answer: the distance field reduces to a native frozen Conv1x1 (W=−2B, bias=‖b‖²), but the argmin/top-k selection stays host-side on Coral/Hailo (RKNN k=1/ArgMin maybe on-device, TopK not); int8+PQ bank fits but costs ~9 pp img-AUROC; no prior art deploys the lookup on a fixed-function NPU.
- RKNN result is v1.6.0-specific; op support is version-dependent and may have changed in newer Toolkit2 releases.

## Sources

- [S1] TensorFlow models on the Edge TPU — coral.ai/docs/edgetpu/models-intro (via gweb-coral-full mirror) — accessed 2026-07-08 (supported-ops table, mandatory full-int8, single-partition CPU fallback, FullyConnected/Concatenation limits)
- [S2] Edge TPU compiler / on-chip 8 MiB caching + off-chip streaming — coral.ai docs + google-coral/edgetpu issues #427/#572 — accessed 2026-07-08
- [S3] Hailo Dataflow Compiler (ONNX→optimize/int8→HEF; Reshape-between-Conv-Dense unsupported, end_node workaround) — developer.ridgerun.com Hailo DFC wiki + DFC User Guide v3.27/v3.30 — accessed 2026-07-08
- [S4] RKNN-Toolkit2 OP Support v1.6.0 (Gather/ArgMax/ArgMin supported; TopK "Not Supported"; no cdist/NN) — github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/RKNN-Toolkit2_OP_Support-1.6.0.md — accessed 2026-07-08
- [S5] Coral 8 MiB SRAM scratchpad + segment streaming — google-coral/edgetpu issue #427; Wikipedia "Tensor Processing Unit" — accessed 2026-07-08
- [S6] Hailo-8 (26 TOPS) / Hailo-8L (13 TOPS), all-memory-on-die / DRAM-free — hailo.ai product pages (hailo-8-ai-accelerator; hailo-8l-ai-accelerator) — accessed 2026-07-08
- [S7] Hailo unsupported-layer handling = truncate at last CONV2D, run pre/post on host — Hailo Community + docs.ultralytics.com/integrations/hailo — accessed 2026-07-08
- [S8] TensorRT mixed precision (FP32/FP16/BF16/FP8/INT8/FP4/INT4), full ONNX parser, TopK layer (K≤3840), custom plugins — docs.nvidia.com/deeplearning/tensorrt + onnxruntime.ai TensorRT EP — accessed 2026-07-08
- [S9] Hailo-8 Datasheet Rev 1.9.2 (Apr 2026) — hailo.ai/hailo-files/hailo-8-datasheet-en — NOT fetched (portal)
- [S10] Hailo-8L Datasheet Rev 1.9.3 (Apr 2026) — hailo.ai/hailo-files/hailo-8l-industrial-datasheet-en — NOT fetched (portal)
