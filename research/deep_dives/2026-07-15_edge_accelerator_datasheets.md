# Edge AI Accelerator Datasheet Specs — Primary-Source Peak TOPS and Memory

**Date**: 2026-07-15
**Status**: partial
**Supersedes**: (none)

## TL;DR

Coral Edge TPU: 4 TOPS int8, 8 MiB SRAM. Hailo-8: 26 TOPS int8, on-die SRAM size not published. Hailo-8L: 13 TOPS int8, on-die SRAM size not published. RK3588 RKNN: 6 TOPS int8, ~1 MB dedicated SRAM + shared LPDDR. Jetson Orin: sparse int8 100–170 TOPS, shared LPDDR 8–64 GB. Raspberry Pi 5: no int8 TOPS spec (CPU); LPDDR4X 1–16 GB.

## Question

For each commodity edge accelerator (Coral Edge TPU, Hailo-8/8L, Rockchip RKNN, NVIDIA Jetson Orin variants) and Raspberry Pi 5: what is the exact peak int8 TOPS performance and on-chip vs. external memory configuration? Return raw facts with primary-source citations only.

## Findings

### 1. Google Coral Edge TPU (USB Accelerator / Dev Board / Modules)

| Spec | Value | Source |
|------|-------|--------|
| **Peak int8 TOPS** | 4 TOPS | [S1] — "4 TOPS peak performance (int8)" |
| **On-chip memory (SRAM)** | 8 MiB | [S2] — "8MB of SRAM... compiler-assigned scratchpad memory" |
| **Memory type** | Scratchpad (not cache) | [S2] — "compiler inserts executable file... writes model parameter data into Edge TPU RAM" |
| **Power envelope** | 2 W | [S1] — "2 watts of power" |

### 2. Hailo-8 AI Accelerator

| Spec | Value | Source |
|------|-------|--------|
| **Peak int8 TOPS** | 26 TOPS | [S3] — "26 TOPS (INT8)" |
| **On-die SRAM capacity** | **Not published** | [S4] — Community discussion, user asked; no official capacity figure in public sources |
| **Memory type** | Integrated on-die, DRAM-free | [S3] — "integrat[es] all required memory directly on the processor die" |
| **External memory** | None (DRAM-free) | [S3] |

**Note**: Hailo datasheets (S5, S6) are available but content not publicly text-extracted; on-die SRAM/memory capacity specification is not disclosed in accessible vendor documentation or community discussions.

### 3. Hailo-8L AI Accelerator

| Spec | Value | Source |
|------|-------|--------|
| **Peak int8 TOPS** | 13 TOPS | [S7] — "up to 13 tera-operations per second (TOPS)" |
| **On-die memory capacity** | **Not published** | [S5], [S6] — Datasheets exist but capacity not disclosed in public sources |
| **Memory type** | Integrated on-die, DRAM-free | [S7] — "Low-Cost DRAM-Free Edge AI Chip" |
| **Power envelope** | 6.6 W TDP | [S8] — Hailo-8L M.2 ET module datasheet |

**Note**: Hailo-8L on-die SRAM/memory capacity specification not available in public datasheets; size relationship to Hailo-8 (roughly half the TOPS) suggests roughly proportional memory, but not confirmed.

### 4. Rockchip RK3588 RKNN NPU

| Spec | Value | Source |
|------|-------|--------|
| **Peak int8 TOPS** | 6 TOPS | [S9] — "6 TOPS of computing power" |
| **Dedicated NPU SRAM** | ~1 MB total; 956 KB usable per IP | [S10] — "RK3588 SOC contains 1MB of SRAM, of which 956KB can be used by each IP" |
| **External memory** | Shared system LPDDR4/4X/LPDDR5 | [S10] — "Dynamic Memory Interface compatible with LPDDR4/LPDDR4X/LPDDR5" |
| **Architecture** | Triple NPU cores (independent or co-work) | [S9] — "triple NPU cores" |

**Note**: RK3588 NPU does not have dedicated DRAM; it uses host system LPDDR via shared bus. SRAM is a constrained on-die resource (1 MB per SoC, ~956 KB per accelerator per docs).

### 5. NVIDIA Jetson Orin Series

#### 5a. Jetson Orin Nano

| Spec | Value | Source |
|------|-------|--------|
| **Peak int8 TOPS (sparse)** | 67 TOPS | [S11] — "67 TOPS of AI performance" |
| **Shared LPDDR memory** | 4 GB or 8 GB | [S12] — NVIDIA spec sheet lists 4GB and 8GB variants |
| **Memory type** | LPDDR5X shared with CPU | [S12] |

#### 5b. Jetson Orin NX

| Spec | Value | Source |
|------|-------|--------|
| **Peak int8 TOPS (sparse)** | 100 TOPS | [S13] — "Compute performance up to 100 (S) INT8 TOPs" |
| **Peak int8 TOPS (dense)** | 50 TOPS | [S13] — "50 (D) INT8 TOPs" |
| **Shared LPDDR memory** | 8 GB or 16 GB | [S12] — NVIDIA spec sheet |
| **Memory type** | LPDDR5X shared with CPU | [S12] |

#### 5c. Jetson AGX Orin

| Spec | Value | Source |
|------|-------|--------|
| **Peak int8 TOPS (GPU sparse)** | 170 TOPS | [S14] — "170 Sparse TOPs of INT8 Tensor compute" (64GB variant) |
| **Peak int8 TOPS (DLA)** | 105 TOPS | [S14] — "105 INT8 Sparse TOPs" on DLAs |
| **Shared LPDDR memory** | 32 GB or 64 GB | [S12], [S14] — NVIDIA spec sheet, 64GB and 32GB variants |
| **Memory type** | LPDDR5X shared with CPU | [S12] — 204 GB/s bandwidth (64GB variant) |

**Note**: Sparse TOPS are higher than dense TOPS due to structured sparsity support (2:4 pattern). Figures quoted are peak; actual int8 inference depends on model sparsity pattern matching.

### 6. Raspberry Pi 5

| Spec | Value | Source |
|------|-------|--------|
| **Peak int8 TOPS (CPU)** | **Not published** | [S15] — CPU specifications available; no vendor int8 TOPS spec. Cortex-A76 is not an AI accelerator. |
| **Processor** | Quad-core ARM Cortex-A76 @ 2.4 GHz | [S15] — BCM2712 SoC |
| **LPDDR memory options** | 1 GB, 2 GB, 4 GB, 8 GB, or 16 GB | [S16] — "LPDDR4X-4267 SDRAM available in 1GB, 2GB, 4GB, 8GB, or 16GB options" |
| **Memory type** | LPDDR4X-4267, shared with CPU | [S16] |

**Note**: Raspberry Pi 5 is a general-purpose CPU/GPU system, not an AI accelerator. No int8 TOPS specification exists (would be measured in integer throughput per the CPU ISA, not neural-network-specific TOPS).

## Open questions

- Hailo-8 and Hailo-8L on-die SRAM/memory capacity in MiB: **datasheets exist but specifications are not accessible in public vendor documentation or community forums.** Hailo may restrict this to sales/evaluation contacts.
- Jetson Orin memory bandwidth specifications (per variant) — only AGX 64GB quoted (204 GB/s) in sources found.
- Raspberry Pi 5 + Coral Edge TPU or Hailo-8 edge-deployed int8 latency measurements (datasheet TOPS do not reflect end-to-end inference time).
- Whether Hailo-8/8L on-die memory capacity has been published in a revision after Rev 1.9.2 / 1.9.3 (April 2026).

## Sources

- [S1] Google Coral Edge TPU — "4 TOPS peak performance (int8), 2 watts of power" — [Coral Accelerator Module datasheet](https://www.coral.ai/static/files/Coral-Accelerator-Module-datasheet.pdf), [USB Accelerator datasheet](https://www.coral.ai/static/files/Coral-USB-Accelerator-datasheet.pdf) (coral.ai, accessed 2026-07-15)
- [S2] Coral Edge TPU 8 MiB SRAM scratchpad — "8MB of SRAM... compiler-assigned scratchpad memory" — [Coral SRAM hierarchy](https://github.com/google-coral/edgetpu/issues/506), Medium article "[What's inside the Google Coral Edge TPU?](https://medium.com/@ilyakavalerov/whats-inside-the-google-coral-edge-tpu-speed-test-teardown-8673b440141d)" (accessed 2026-07-15)
- [S3] Hailo-8 26 TOPS and DRAM-free architecture — "26 TOPS (INT8)" and "integrates all required memory directly on the processor die" — [Hailo-8 AI Accelerator product page](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/), [Hailo-8 M.2 AI Acceleration Module](https://hailo.ai/products/ai-accelerators/hailo-8-m2-ai-acceleration-module/) (hailo.ai, accessed 2026-07-15)
- [S4] Hailo-8 on-die SRAM capacity not published — [Hailo Community discussion: "hailo-8 AI accelerator chip, how much SRAM does it have?"](https://community.hailo.ai/t/hailo-8-ai-accelerator-chip-how-much-sram-does-it-have/19111) (hailo.ai community, accessed 2026-07-15)
- [S5] Hailo-8 Datasheet Revision 1.9.2 (April 2026) — [hailo.ai/hailo-files/hailo-8-datasheet-en/](https://hailo.ai/hailo-files/hailo-8-datasheet-en/) (portal access; content not text-extracted)
- [S6] Hailo-8L Datasheet Revision 1.9.3 (April 2026) — [hailo.ai/hailo-files/hailo-8l-industrial-datasheet-en/](https://hailo.ai/hailo-files/hailo-8l-industrial-datasheet-en/) (portal access; content not text-extracted)
- [S7] Hailo-8L 13 TOPS and DRAM-free — "up to 13 tera-operations per second (TOPS)" — [Hailo-8L AI Accelerator product page](https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-for-ai-light-applications/) (hailo.ai, accessed 2026-07-15)
- [S8] Hailo-8L M.2 ET Module 6.6 W TDP — [Hailo-8L M.2 ET Module (Key A+E / Key B+M) datasheets](https://hailo.ai/hailo-files/hailo-8l-m-2-key-a-e-et-datasheet-en/) (hailo.ai, accessed 2026-07-15)
- [S9] RK3588 RKNN NPU 6 TOPS, triple-core architecture — "6 TOPS... triple NPU cores" — [Rockchip RK3588 Specifications](https://www.cnx-software.com/2020/11/26/rockchip-rk3588-specifications-revealed-8k-video-6-tops-npu-pcie-3-0-up-to-32gb-ram-/) (CNX Software, 2020), [RK3588 NPU performance analysis](https://www.accio.com/plp/rockchip-rk3588-npu-tops-specifications) (accessed 2026-07-15)
- [S10] RK3588 SRAM and LPDDR memory architecture — "RK3588 SOC contains 1MB of SRAM, of which 956KB can be used by each IP" and "Dynamic Memory Interface compatible with LPDDR4/LPDDR4X/LPDDR5" — [rknpu2 / RK3588_NPU_SRAM_usage.md](https://github.com/rockchip-linux/rknpu2/blob/master/doc/RK3588_NPU_SRAM_usage.md), [RK3588 Datasheet V1.6](https://wiki.friendlyelec.com/wiki/images/e/ee/Rockchip_RK3588_Datasheet_V1.6-20231016.pdf) (GitHub/Rockchip, accessed 2026-07-15)
- [S11] Jetson Orin Nano 67 TOPS — "67 TOPS of AI performance" — [NVIDIA Jetson Orin Nano product page](https://developer.nvidia.com/embedded/jetson-modules) (nvidia.com, accessed 2026-07-15)
- [S12] Jetson Orin memory variants (4/8/16/32/64 GB LPDDR5X) — [NVIDIA Jetson Orin modules spec sheet](https://developer.nvidia.com/embedded/jetson-modules), Jetson Orin NX and AGX product briefs (developer.nvidia.com, accessed 2026-07-15)
- [S13] Jetson Orin NX 100 Sparse / 50 Dense int8 TOPS — "Compute performance up to 100 (S) INT8 TOPs and 50 (D) INT8 TOPs" — [Jetson Orin NX Technical Brief](https://developer.nvidia.com/downloads/jetson-orin-nx-series-data-sheet), NVIDIA Developer Forums "[How to verify Orin the TOPS performance](https://forums.developer.nvidia.com/t/how-to-verify-orin-the-tops-performance/305262)" (developer.nvidia.com, accessed 2026-07-15)
- [S14] Jetson AGX Orin 170 GPU Sparse / 105 DLA int8 TOPS, 32/64 GB LPDDR — "170 Sparse TOPs of INT8 Tensor compute" and "105 INT8 Sparse TOPs" on DLAs — [NVIDIA Jetson AGX Orin Series Technical Brief v1.2](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf), NVIDIA Developer Forums "[Jetson AGX Orin TOPs / CUDA Cores Explained](https://forums.developer.nvidia.com/t/jetson-agx-orin-tops-cuda-cores-explained/252426)" (nvidia.com, accessed 2026-07-15)
- [S15] Raspberry Pi 5 BCM2712 Cortex-A76 @ 2.4 GHz, no int8 TOPS — "quad-core 64-bit Arm Cortex-A76 processor... clocked at 2.4GHz" — [Raspberry Pi Documentation: Processors](https://www.raspberrypi.com/documentation/computers/processors.html), [Raspberry Pi 5 Product Brief](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf) (raspberrypi.com, accessed 2026-07-15)
- [S16] Raspberry Pi 5 LPDDR4X memory options — "LPDDR4X-4267 SDRAM available in 1GB, 2GB, 4GB, 8GB, or 16GB options" — Waveshare Raspberry Pi 5 specs, Micro Center product listings (accessed 2026-07-15)
