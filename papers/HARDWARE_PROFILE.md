# WEFT Certified Physical Hardware Execution Profile

This document details the exact, machine-verified hardware characteristics and operating system topology of the certified testbed supporting all experiments, SIMD kernels, and GPU dispatch pipelines in WEFT.

---

## 1. Physical Host System Profile (AMD Ryzen 5 7600X)

- **Architecture:** AMD Zen 4 (64-bit x86_64)
- **Cores / Threads:** 6 Physical Cores / 12 Hardware Threads (SMT enabled)
- **Base / Boost Clocks:** 4.70 GHz base / up to 5.35 GHz boost
- **L1 Cache:** 384 KB (6 × 32 KB I-Cache + 6 × 32 KB D-Cache)
- **L2 Cache:** 6 MB (6 × 1 MB private per core)
- **L3 Cache:** 32 MB unified shared cache
- **Main Memory:** 32 GB DDR5-5200 Dual-Channel
- **AVX-512 Instruction Set Extensions:**
  - `AVX-512F` (Foundation)
  - `AVX-512DQ` (Doubleword and Quadword Operations)
  - `AVX-512BW` (Byte and Word Operations)
  - `AVX-512VL` (Vector Length Extensions: 128, 256, and 512-bit registers)
  - `AVX-512VNNI` (Vector Neural Network Instructions)
  - `AVX-512VPOPCNTDQ` (Vector Population Count)
  - `AVX-512BITALG` (Bit Manipulation)

---

## 2. Hardware Accelerator Profile (NVIDIA GeForce RTX 5060 Ti)

Queried live via `nvidia-smi` and `torch.cuda.get_device_properties(0)`:
- **Product Name:** NVIDIA GeForce RTX 5060 Ti
- **Microarchitecture:** Blackwell
- **Compute Capability (SM):** 12.0
- **Total Physical VRAM:** 16 GB GDDR7 (15.53 GB accessible)
- **Driver Version:** 620.70.06
- **CUDA Runtime / Driver:** CUDA 12.8 / 13.2+
- **Triton Version:** Triton 3.9
- **Host Interconnect:** PCIe 4.0 x16

---

## 3. Storage Hierarchy & Safety Floors

- **Model Archive Mount:** `/run/media/akarta/AI` (Floor: $\ge 700\text{ GB}$ free)
- **Host System Root:** `/home` (Floor: $\ge 200\text{ GB}$ free)
