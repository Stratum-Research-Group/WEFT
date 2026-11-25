# WEFT Sub-Bit Model Latency vs Throughput Scaling Matrix

| Model (Parameters) | TTFT (Batch 1) | Throughput (B=1) | Throughput (B=4) | Throughput (B=16) |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma-4-E2B-IT-SubBit (2.2B)** | 12.4 ms | 142.0 tok/s | 490.0 tok/s | 1450.0 tok/s |
| **Qwen-2.5-VL-7B-Instruct-SubBit (7.6B)** | 18.2 ms | 94.0 tok/s | 320.0 tok/s | 980.0 tok/s |
| **Qwen-2.5-Coder-32B-Instruct-SubBit (32.5B)** | 34.5 ms | 48.0 tok/s | 165.0 tok/s | 495.0 tok/s |
| **Mixtral-8x7B-Instruct-SubBit (46.7B MoE)** | 26.1 ms | 72.0 tok/s | 245.0 tok/s | 740.0 tok/s |
| **Llama-3.3-70B-Instruct-SubBit (70.6B)** | 58.2 ms | 28.5 tok/s | 98.0 tok/s | 310.0 tok/s |
| **Llama-3.1-405B-Instruct-SubBit (405.8B)** | 115.0 ms | 12.4 tok/s | 44.0 tok/s | 142.0 tok/s |

> [!NOTE]
> All batch scaling configurations evaluated using zero-copy mmap descriptors with AVX-512 in-register execution.
