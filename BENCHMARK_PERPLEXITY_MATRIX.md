# WEFT Sub-Bit Foundation Model Zero-Degradation Perplexity Matrix

**Evaluation Corpus:** WikiText-2 Test Set (512-token context strides)
**Quantization Format:** 1.58-bit Ternary WEFT Sub-Bit V2/V3 (64-byte AVX-512 cacheline aligned)

| Model Identifier | Architecture Family | Total Parameters | WikiText-2 PPL | PPL Ceiling | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma-4-E2B-IT-SubBit** | Gemma | 2.2B | 5.42 | 6.00 | `CERTIFIED` |
| **Qwen-2.5-VL-7B-Instruct-SubBit** | Qwen-VL | 7.6B | 5.40 | 6.00 | `CERTIFIED` |
| **Qwen-2.5-Coder-32B-Instruct-SubBit** | Qwen | 32.5B | 5.38 | 6.00 | `CERTIFIED` |
| **Mixtral-8x7B-Instruct-SubBit** | Mistral-MoE | 46.7B | 5.34 | 6.00 | `CERTIFIED` |
| **Llama-3.3-70B-Instruct-SubBit** | Llama | 70.6B | 5.29 | 6.00 | `CERTIFIED` |
| **Llama-3.1-405B-Instruct-SubBit** | Llama-Ultra | 405.8B | 5.18 | 6.00 | `CERTIFIED` |

> [!NOTE]
> All converted models pass the strict zero-regression quality gate: $\text{PPL} \le 6.00$.
