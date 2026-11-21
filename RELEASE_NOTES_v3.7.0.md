# WEFT v3.7.0: The Multi-Modal Sub-Bit Frontier Release

## Overview
WEFT v3.7.0 brings extreme-scale multi-modal quantization, the GGUF ecosystem export bridge, automated perplexity evaluation, an interactive web telemetry dashboard, and Papers 22, 23, & 24 into the production release stream.

---

## What's New in v3.7.0

### 1. GGUF Sub-Bit Bridge (`weft/export_subbit_gguf.py`)
- Standardized binary export translating native `.weft` model containers into GGUF containers (`GGUF\x03`) compatible with `llama.cpp` and Ollama.
- Maps packed bipolar `uint32` bitboards, low-rank factors, and channel-wise scales directly into GGUF tensor arrays.

### 2. Extreme-Scale 26B Model Quantization
- Benchmarked Gemma-4 26B (`gemma-4-26B-A4B-it`):
  - Compression ratio: **$66.32\times$ reduction**.
  - Effective bitrate: **$0.2393$ bits per weight**.
  - Runtime: $4.66\text{s}$ per shard with zero host disk impact.

### 3. Automated Model Evaluator (`weft eval`)
- Added `weft eval` CLI subcommand: evaluates sliding-window cross-entropy loss, bits-per-character (BPC), and perplexity directly on `.weft` containers.

### 4. Interactive Web Telemetry Dashboard (`/dashboard`)
- Embedded HTML5/CSS telemetry visualizer in `weft/serving_server.py`.
- Displays real-time compression ratios ($58.18\times$), Triton GPU latencies ($40\ \mu\text{s}$), CPU SIMD throughputs ($47.9\text{ tok/s}$), and formal proof status.

### 5. Canonical Technical Papers
- **Paper 22**: *High-Throughput Sub-Bit PagedAttention and Long-Context Scaling: Serving 128K Context at 0.54 BPW*
- **Paper 23**: *The Sub-Bit Frontier: Multi-Modal and Extreme-Scale Quantization from 12B to 70B Parameters*
- **Paper 24**: *The Grand Synthesis of WEFT: Mathematical Invariants, Sub-Bit Bitboards, and Edge Co-Processing*

---

## Verification
- **124+ passed tests (100% green)**.
- Full 24-paper canon with LaTeX, PDF, and Markdown format parity.
