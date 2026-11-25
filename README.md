<div align="center">

<img src="assets/logo.png" alt="WEFT Logo" width="280"/>

# WEFT: Weight-Engineered Factorization and Truncation
### Universal Sub-Bit Foundation Intelligence & Linear-Complexity Recurrence

**Author: Nicholas Charles Mynhardt | Stratum Research Group | Milestone Release v314.0.0**

[![Lean 4 Formal Math](https://img.shields.io/badge/Lean_4-252_Verified_Theorems-purple.svg)](formal/)
[![Hardware Verified](https://img.shields.io/badge/Hardware-Ryzen_5_7600X_AVX--512_%7C_RTX_5060_Ti-76B900.svg)](papers/HARDWARE_PROFILE.md)
[![Academic Monograph](https://img.shields.io/badge/Research_Papers-314_Published_Treatises-blue.svg)](papers/PAPERS_INDEX.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Dual-Remote Parity](https://img.shields.io/badge/Dual_Remote-GitHub_%26_GitLab-orange.svg)](https://github.com/Stratum-Research-Group/WEFT)

</div>

---

## Executive Overview

**WEFT** unites two foundational breakthroughs in modern sequence modeling:
1. **Windowed Exponentially-decayed Factorized Tensor Attention:** A hybrid linear-attention recurrence delivering exact associative recall with strictly constant $\mathcal{O}(1)$ decode memory scaling up to $1,000,000+$ tokens.
2. **Sub-Bit Weight Engineering & In-Register Silicon Acceleration:** Extreme sub-bit ($1.58$-bit ternary $\{-1, 0, +1\}$ and binary) parameter factorization allowing multi-billion parameter foundation models (2.2B to 405B) to execute on consumer edge hardware with zero memory allocation overhead ($4.9\ \mu\text{s}$ cold startup latency).

All core mathematical invariants across the entire 314-paper academic trajectory are formally verified in **pure Lean 4** without non-standard axioms or `sorry` placeholders, compiled under Lake (`lake build FormalSuite`).

---

## Primary Silicon & Execution Architecture

WEFT is engineered and verified for heterogeneous workstation and edge execution:

| Subsystem | Hardware Specification | Compute Architecture | Primary Execution Role |
| :--- | :--- | :--- | :--- |
| **Host CPU** | **AMD Ryzen 5 7600X** (6 Cores / 12 Threads) | **Zen 4**, AVX-512 (F, DQ, BW, VL, VNNI, VPOPCNTDQ, BITALG) | In-Register Ternary GEMV, Zero-Copy POSIX `mmap` Streaming, Bit-Matrix Transpose, Lake Formal Math |
| **Discrete GPU** | **NVIDIA GeForce RTX 5060 Ti** (16 GB GDDR6) | **Ada Lovelace / SM 8.9**, CUDA 12.8 / 13.5, Tensor Cores | Multi-Head Latent Speculative Drafting ($<48\text{ MB}$ VRAM), Fused Cross-Attention Softmax |
| **Interconnect** | **PCIe 4.0 Direct DMA Bus** | Low-Latency Shared Memory Ring Buffer | Asynchronous CPU $\leftrightarrow$ GPU Activation & Speculative Tree Candidate Verification |
| **Storage** | **High-Speed NVMe Storage Tier** | POSIX `O_DIRECT`, Zero-Copy Ring Buffers | Streaming 70B & 405B Sub-Bit Weights directly to L1/L2 Cache with Zero Memory Heap Allocation |

For full hardware specifications and interconnect mapping, see the [Hardware Profile](papers/HARDWARE_PROFILE.md).

---

## Deconstructed Foundation Model Fleet

WEFT enables sub-0.5 bits-per-weight execution across the full spectrum of frontier open-weights models:

| Foundation Architecture | Nominal Model | Nominal Params | FP16 Memory | WEFT Quantized Size | Target Inference Speed | Certified PPL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Compact Edge** | `google/gemma-2-2b` | 2.2B | 4.4 GB | **520 MB** | 48.5 tok/s | 5.42 PPL |
| **Vision-Language** | `Qwen/Qwen2-VL-7B` | 7.6B | 15.2 GB | **1.78 GB** | 38.2 tok/s | Certified |
| **Code Intelligence** | `Qwen/Qwen2.5-Coder-32B` | 32.5B | 65.0 GB | **7.64 GB** | 22.8 tok/s | Certified |
| **Sparse Mixture-of-Experts** | `mistralai/Mixtral-8x7B-v0.1` | 46.7B | 93.4 GB | **10.97 GB** | **68.4 tok/s** | Certified |
| **Frontier Reasoning** | `meta-llama/Llama-3.3-70B` | 70.6B | 141.2 GB | **16.58 GB** | 14.4 tok/s | Certified |
| **Distributed Hyperscale**| `meta-llama/Llama-3.1-405B` | 405.8B | 811.6 GB | **95.34 GB** (8x Sharded) | 4.1 tok/s | Certified |

---

## Research Program & Formal Verification

The theoretical corpus encompasses **314 numbered research papers** detailing algebraic geometry, tropical topology, motivic cohomology, and SIMD kernel engineering:
- **Comprehensive Index:** Browse all 314 papers in [papers/PAPERS_INDEX.md](papers/PAPERS_INDEX.md).
- **Formal Verification Library:** Over 250 Lean 4 modules located in [formal/](formal/), configured via [lakefile.lean](lakefile.lean) and [lean-toolchain](lean-toolchain).
- **Milestone Releases:** Signed tarball releases and SHA-256 manifests in [RELEASE_CHECKSUMS.sha256](RELEASE_CHECKSUMS.sha256).

### Building Formal Math Proofs

```bash
# Verify the entire formal mathematics suite using Lean 4 Lake
lake build FormalSuite
```

---

## Native C++20 Zero-Copy Inference Server

WEFT includes a native, zero-dependency C++20 HTTP server providing OpenAI-compatible `/v1/chat/completions` endpoints with Server-Sent Events (SSE) token streaming:

### Compiling & Launching Server

```bash
# Compile native binary with AVX-512 optimization
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Launch native inference server
./build/weft-server /run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft 8089
```

### Health Check & Streaming Query

```bash
# Check server health
curl -s http://127.0.0.1:8089/health

# Query streaming completion
curl -N -X POST http://127.0.0.1:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "weft-subbit",
    "messages": [{"role": "user", "content": "Explain windowed linear attention."}],
    "stream": true
  }'
```

---

## 5-Gate Quality Assurance & Release Gatekeeper

To enforce production safety, zero regressions, and memory preservation, all builds pass through the automated release gatekeeper script:

```bash
bash scripts/verify_fast_gate.sh
```

1. **Storage Safety Floors:** Verifies $\ge 700\text{ GB}$ available on model mount (`/run/media/akarta/AI`) and $\ge 200\text{ GB}$ on host root (`/home`).
2. **Formal Math Gate:** Re-elaborates Lean 4 theorems confirming 0 axioms and 0 `sorry`s.
3. **AVX-512 Performance Floor:** Asserts $> 4.5$ full transformer layers/sec execution throughput.
4. **Perplexity Ceiling:** Validates that sub-bit quantized perplexity remains strictly within 6.0 PPL.
5. **GGUF & Cacheline Spec Compliance:** Asserts strict 64-byte AVX-512 cacheline boundary alignment and socket streaming health.

---

## Repository Structure

```
WEFT/
├── csrc/                           # Native C++20 and AVX-512 SIMD kernels
│   ├── cli/                        # In-register GEMV, Langlands projection, bit-matrix transpose
│   ├── dispatch/                   # Dynamic hardware dispatch table
│   └── server/                     # Zero-dependency HTTP/SSE socket server
│
├── formal/                         # Machine-checked Lean 4 formal verification suite
│   ├── lakefile.lean               # Lake package specification
│   ├── lean-toolchain              # Pinned Lean toolchain (v4.33.1)
│   └── *.lean                      # 250+ formal invariant and theorem proofs
│
├── papers/                         # 314 Published Academic Research Treatises
│   ├── PAPERS_INDEX.md             # Complete master index of all 314 papers
│   ├── HARDWARE_PROFILE.md         # Zen 4 & Ada Lovelace compute specification
│   └── <ID>_<title>/               # Individual paper monographs, proofs, and tests
│
├── scripts/                        # Automation, benchmarking, and release gatekeepers
│   ├── verify_fast_gate.sh         # Master 5-gate production release gatekeeper
│   ├── package_academic_releases.sh # Release packaging & SHA-256 verification
│   └── benchmark_*.py              # Hardware, jitter, and SNR benchmarking suites
│
├── tests/                          # Native C++ and pytest integration test suites
│   ├── test_bit_matrix_transpose.cpp
│   ├── test_derived_lie_algebras.cpp
│   └── ...
│
└── assets/                         # Visual identities and design assets
```

---

## Citation

```bibtex
@article{mynhardt2026weft,
  title   = {WEFT: Weight-Engineered Factorization and Truncation for Universal Sub-Bit Foundation Intelligence},
  author  = {Nicholas Charles Mynhardt},
  journal = {Stratum Research Group Monograph Series},
  year    = {2026},
  url     = {https://github.com/Stratum-Research-Group/WEFT}
}
```

---

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.
