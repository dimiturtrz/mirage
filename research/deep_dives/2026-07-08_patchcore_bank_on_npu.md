# Can a coreset-reduced, int8 PatchCore memory bank + (A)NN run on a commodity NPU?

**Date**: 2026-07-08
**Status**: settled (with one honest inference flagged)
**Supersedes**: (none)
**Related**: [2026-07-08_edge_accelerator_landscape.md](2026-07-08_edge_accelerator_landscape.md) — establishes the op-support facts this builds on (Coral/Hailo/RKNN op tables, int8-mandatory, 8 MiB / DRAM-free envelopes). Read that first.

## TL;DR

The **matmul half is reducible; the argmin/top-k half is the irreducible host-side residue**. Distance-vs-fixed-bank is genuinely a frozen Conv1x1/FC (weights −2·B, bias ‖b‖²) and maps to the accelerator natively — this is real, not hand-waving. What does NOT map: the argmin/top-k over the bank dimension (Coral: neither in op table; Hailo: not in dataflow graph; RKNN: ArgMax/ArgMin yes but **TopK no**), so **k=1 is closer to on-device-feasible than k>1**, but on Coral/Hailo the reduction still amputates to host at the argmin. Net: a small int8 PQ bank **fits** (334 KB for 10k×256 [S3]) and int8/PQ costs ~**9 pp image-AUROC** (0.95→0.86 [S3]) — but **no one in the literature has actually put the bank lookup on a fixed-function NPU**; every measured edge PatchCore ran the NN on host CPU or Jetson GPU [S3][S4]. Honest answer: **reducible to "backbone+distance-matmul on-NPU, argmin/top-k on host" — not to fully-on-NPU** on Coral/Hailo/RKNN-class.

## Question

Can a coreset-reduced, int8-quantized PatchCore memory bank + (approximate) NN be coerced onto a commodity NPU (Coral/Hailo/RKNN), or is the lookup irreducibly host-CPU? Cover: (1) distance-as-matmul trick + the argmin/top-k residue per accelerator; (2) int8 + memory feasibility; (3) prior art actually deployed on edge; (4) on-device-lookup alternatives (PQ/IVF-ADC, or "just use EfficientAD/AE instead").

## Findings

### 1. The distance-as-matmul trick — correct, and half-native

- **The reformulation is algebraically correct.** For a fixed bank B, argmin_b ‖x−b‖² = argmin_b(‖x‖² − 2·x·b + ‖b‖²). ‖x‖² is constant across b for a given query x, so it drops from the argmin: argmin_b(‖b‖² − 2·x·b). This is exactly a linear layer **W = −2·B, bias = ‖b‖²** (both precomputable and frozen, since B is fixed), followed by argmin over the bank dimension. Corroborated by the field: NN "can be efficiently implemented using matrix multiplication to compute pairwise distances... with argmin then selecting the minimum" [S1], and formally "Maximum Inner Product is Query-Scaled Nearest Neighbor" — the MIPS↔NN equivalence underpinning the −2·x·b term [S5]. (Math verified by inspection; not merely cited.)
- **The matmul part IS accelerator-native.** Expressing B as a frozen Conv1x1/FullyConnected weight is exactly what these NPUs accelerate. On Coral, `FullyConnected` is supported (with 1-D output constraint) [see landscape S1]; on Hailo it's a dense layer in the dataflow graph; on RKNN it's a supported matmul. So "compute the distance field" reduces to a native op — **this is the reducible half.**
- **The argmin/top-k is the residue, and it splits on k:**
  - **k=1** needs a single **argmin** over the bank axis. RKNN lists **ArgMin supported** [landscape S4] → on an RKNN-class NPU, k=1 distance+argmin is *potentially* fully on-NPU (subject to the argmin being over the right axis/shape — not separately verified). On **Coral** argmin/argmax are **not in the op table**, and on **Hailo** there is no data-dependent argmin node in the dataflow graph → **host-side even for k=1** [landscape S1][landscape S7].
  - **k>1** needs **top-k**. **TopK is unsupported on RKNN, absent on Coral, and not a dataflow op on Hailo** [landscape S4] → **host-side on all three.** PatchCore's canonical scoring uses the max over patches of the patch's NN distance and a k>1 re-weighting for the image score, so the deployed form leans toward top-k, not pure k=1.
- **Consequence.** Best case is "**backbone + distance-matmul on-NPU, argmin/top-k on host**." On RKNN k=1 *might* close fully on-device; on Coral/Hailo the graph still truncates at the argmin and the reduction ends there. The matmul reformulation removes the *bank-storage-and-cdist* objection but **not** the *selection* objection on Coral/Hailo.

### 2. Int8 + memory feasibility — the bank fits; accuracy pays ~9 pp

- **A coreset+int8+PQ bank fits the envelope comfortably.** PatchCore-Lite: coreset to **K=10000, d=256**, product-quantised (m=8 subspaces, b=8 bits → 256 centroids/subspace), giving **~334 KB vs the original 9.77 MB** [S3]. 334 KB is far inside Coral's 8 MiB SRAM and any Hailo on-die budget — **memory is not the blocker.** (Even a raw int8 10k×256 bank = 2.5 MB, still inside 8 MiB.)
- **Int8/PQ quantization of the bank measurably degrades anomaly accuracy — non-trivially at image level:** PatchCore-Lite image-level **I-ROC 0.95 → 0.86** (~9 pp drop) and pixel-level **P-ROC 0.96 → 0.94** (~2 pp) under PQ [S3]; PaDiM-Lite similarly 0.94 → 0.85 image / 0.97 → 0.94 pixel [S3]. So the *localization* survives quantization well but the *image-level detection* takes a real hit. This is PQ-approximation-dominated, not purely int8 rounding, but it's the cost of shrinking the bank to edge size. (Contrast: generic int8 PTQ on *classification* loses <1 pp [S2] — anomaly image-scoring is more fragile.)

### 3. Prior art — the lookup is always on host CPU or Jetson GPU, never on a fixed-function NPU

- **PatchCore-Lite / PaDiM-Lite (the closest prior art): CPU only.** "All experiments... conducted on a Intel Core i5-9600K CPU... to simulate edge deployment" [S3]. The two-stage PQ search (coarse on PQ bank → exact on decoded subset [S3]) ran on CPU; **no Coral/Hailo/NPU deployment.**
- **Continual-AD-on-edge benchmark: Jetson GPU + RPi5 CPU only, no fixed-function NPU.** PatchCore 119 ms / 8.4 FPS (AUROC 0.925); EfficientAD 34.7 ms / 28.8 FPS but AUROC collapses to 0.529 under their continual setting; DINOSaur 91.6 ms / 0.968 [S4]. Explicitly: "**no methods were deployed on fixed-function NPUs (Coral, Hailo)**" — Jetson Orin Nano GPU and RPi5 CPU only [S4].
- **Edge Impulse ships PatchCore "MCU to NPU"** but the public blog gives **no evidence the NN lookup runs on the NPU** vs the host MCU/CPU — it defers technical detail to docs [S6]. Treat as unconfirmed for on-NPU lookup.
- **Inference (flagged):** across the surveyed literature, the *backbone* is what gets accelerated; the *bank NN* is consistently a host operation. I found **no paper demonstrating the argmin/top-k tail running on a Coral/Hailo/RKNN NPU.** Absence-of-evidence, but consistent with the op-table constraints in the landscape deep-dive.

### 4. Alternatives that keep the lookup on-device

- **PQ / IVF-ADC as fixed ops:** PQ replaces exact cdist with **precomputed distance lookup tables** (prototypes×query computed, then table gather-and-sum) [from PQ refs]. This trades the matmul for **gather+add over LUTs** — but gather-heavy table scans are **CPU-bound even on CPUs** (ADC scans at 2–3 GB/s, memory-bound [S7-pq]) and **`Gather` is only sometimes supported** (RKNN yes; Coral no) — so IVF-ADC does *not* obviously map better than the matmul form; it moves the residue from top-k to gather, which is equally host-ish on Coral/Hailo.
- **NPU-side ANN exists but on big/heterogeneous NPUs, not Coral-class:** Ascend-RaBitQ does billion-scale ANN with 1-bit quantization on an **Ascend NPU + CPU heterogeneously** [S8] — i.e. even the state of the art keeps a CPU in the loop and targets a datacenter-class NPU, not an 8 MiB EdgeTPU. Confirms the selection step wants a general processor.
- **The field's actual edge-anomaly answer is "use a distillation/reconstruction model instead of a bank."** EfficientAD (student-teacher + autoencoder) and conv-AE approaches are **pure-conv, fully int8-quantizable, no lookup** → they map cleanly onto *all* the commodity NPUs (per landscape). The tradeoff is accuracy/robustness: EfficientAD is ~3–4× faster (34.7 vs 119 ms [S4]) but its detection can collapse under distribution shift / continual settings (0.529 in [S4]), whereas PatchCore holds 0.925. So the honest framing: **bank-free conv models are the "runs-entirely-on-NPU" path, at a robustness cost the reader must weigh** — this is the same conclusion the landscape deep-dive reached from the op tables, now corroborated by measured edge numbers.

## Open questions

- Does RKNN's supported `ArgMin` actually accept the shape/axis a k=1 PatchCore distance field produces (large bank axis)? Op *listed* ≠ op *usable at this shape* — unverified.
- Real on-NPU latency of the distance-matmul for a 10k×256 int8 bank (a 10k-wide FC) on Coral/Hailo — is the FC itself cheap enough to matter, or does it blow the SRAM/dataflow budget? Not measured anywhere found.
- Whether Edge Impulse's "PatchCore to NPU" actually offloads the NN or only the backbone — needs their docs, not the blog [S6].
- Does QAT (vs the PTQ/PQ in [S3]) recover the ~9 pp image-AUROC drop? [S2] hints QAT helps detection/segmentation; untested for PatchCore banks.

## Sources

- [S1] PatchCore NN = matmul pairwise-distance + argmin (algorithm description) — anomalib docs + dataroots PatchCore blog + "Towards Total Recall" (arxiv 2106.08265) — accessed 2026-07-08
- [S2] Generic int8 PTQ loses <1 pp on classification; detection/segmentation may need QAT — TechnoLynx IoT Edge AI deploy guide + OpenReview "Bridging AI Quantization and Edge" — accessed 2026-07-08
- [S3] **PatchCore-Lite / PaDiM-Lite** — "Efficient Visual Anomaly Detection at the Edge" arxiv 2603.20288 (html) — PQ bank m=8/b=8, K=10000 d=256, 334 KB vs 9.77 MB; I-ROC 0.95→0.86, P-ROC 0.96→0.94 (PatchCore-Lite); CPU-only (i5-9600K) — accessed 2026-07-08
- [S4] "Rethinking Continual Anomaly Detection on the Edge" arxiv 2605.24251 — Jetson Orin Nano: PatchCore 119 ms/8.4 FPS/0.925, EfficientAD 34.7 ms/28.8 FPS/0.529, DINOSaur 91.6 ms/0.968; no fixed-function NPU deployed (Jetson GPU + RPi5 CPU only) — accessed 2026-07-08
- [S5] "Maximum Inner Product is Query-Scaled Nearest Neighbor" — VLDB vol18 p1770 — MIPS↔NN equivalence (the −2·x·b linearization) — accessed 2026-07-08
- [S6] Edge Impulse "PatchCore Boosts Visual Anomaly Detection" blog — claims MCU→NPU→GPU support; no evidence NN lookup runs on-NPU (defers to docs) — accessed 2026-07-08
- [S7-pq] PQ/IVF-ADC background: ADC is CPU/memory-bound (2–3 GB/s scan); LUT gather-and-sum — Jégou PQ (researchgate 47815472) + arxiv 1712.02912 — accessed 2026-07-08
- [S8] "Ascend-RaBitQ: Heterogeneous NPU-CPU Acceleration of Billion-Scale Similarity Search" arxiv 2605.16007 — even SOTA NPU-ANN is NPU+CPU heterogeneous on a datacenter-class NPU — accessed 2026-07-08
- [landscape S1/S4/S7] op-support facts (Coral no argmin/topk; RKNN ArgMin yes / TopK no; Hailo dataflow-only, truncate-at-conv) — see 2026-07-08_edge_accelerator_landscape.md
