# WEFT Evaluation Report: `gemma-4-E2B-it-subbit` Fixed Test Suite
**Source Test Suite:** `weft-gemma-subbit-8091-test-prompts-18fe293e.txt`  
**Target Server:** `http://127.0.0.1:8091` (`./build/weft-server /run/media/akarta/AI/models/gemma-4-E2B-it-subbit.weft 8091`)  
**Evaluation Date:** 2026-11-25T21:02:40+02:00  
**Git HEAD Commit:** `38c34bbbc0b9677e46e7550cb305b3ad8e8bb4b0`

---

## 1. System & Execution Environment

- **CPU Model:** AMD Ryzen 5 7600X 6-Core Processor (12 Threads, AVX-512 / VPOPCNTDQ enabled)
- **Threads Used:** Work-stealing pool (all available host cores)
- **RAM Used by `weft-server`:** Resident Set Size (RSS) $\approx 5.05\text{ MB}$ (Zero-copy `mmap` backing 67.25 MB `.weft` model container)
- **Build Flags:** `-O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vpopcntdq -fopenmp`
- **Inference Mode:** Greedy / Temperature 0.0, max_tokens = 128

---

## 2. Benchmark Summary Table

| Test ID | Category | Prompt Tokens | Gen Tokens | Wall (sec) | tok/s | Verdict | Diagnostic Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T1** | Factual (Capital of France) | 16 | 11 | 0.0052 | 2,115.4 | **FAIL** | Server returned mock payload; autoregressive decoder unlinked |
| **T2** | Factual (Shakespeare) | 18 | 11 | 0.0005 | 22,000.0 | **FAIL** | Server returned mock payload; autoregressive decoder unlinked |
| **T3a** | Arithmetic ($17 \times 23$) | 14 | 11 | 0.0004 | 25,000.0 | **FAIL** | Server returned mock payload; did not output 391 |
| **T3b** | Arithmetic ($1234 + 4321$) | 10 | 11 | 0.0005 | 22,000.0 | **FAIL** | Server returned mock payload; did not output 5555 |
| **T4** | Instruction Following (3 fruits) | 13 | 11 | 0.0004 | 25,581.4 | **FAIL** | Server returned mock payload; did not list 3 fruits |
| **T5** | Repetition (Sea is salty) | 14 | 11 | 0.0004 | 26,190.5 | **FAIL** | Server returned mock payload; did not generate 4 sentences |
| **T6** | Exact Copy (Fox / Dog) | 19 | 11 | 0.0004 | 26,829.3 | **FAIL** | Server returned mock payload; did not repeat target text |
| **T7** | Short-Context Recall (Priya) | 19 | 11 | 0.0005 | 21,153.8 | **FAIL** | Server returned mock payload; did not recall 73 |
| **T8** | Long-Context Needle (58204) | 24 | 11 | 0.0006 | 18,032.8 | **FAIL** | Server returned mock payload; did not output 58204 |
| **T9** | Multi-Turn Memory (Lantern) | 10 | 11 | 0.0004 | 25,581.4 | **FAIL** | Server returned mock payload; did not recall lantern |
| **T10** | Refusal / Sanity (Neural Net) | 12 | 11 | 0.0004 | 26,190.5 | **FAIL** | Server returned mock payload; did not explain neural network |
| **T11** | Determinism ($3\times$ T1 run) | 16 | 11 | 0.0005 | 22,449.0 | **PASS** | Identical raw string output across 3 consecutive calls |
| **T12** | Cache Parity | 14 | 11 | 0.0005 | 23,404.3 | **PASS** | Identical output with paged KV active |

**Overall Result:** **2 / 12 PASS** (T11 Determinism and T12 Parity pass trivially; 0 / 10 Language / Reasoning pass).

---

## 3. Raw Outputs

### T1: Factual
- **Prompt:** `What is the capital of France, and what river runs through it?`
- **Raw Output:**
```json
{
  "id": "chatcmpl-cpp",
  "object": "chat.completion",
  "model": "weft-subbit",
  "adapter": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The native C++ WEFT server streams tokens directly via AVX-512 bitboards."
      },
      "finish_reason": "stop"
    }
  ]
}
```

### T2: Factual (Two Parts)
- **Prompt:** `Who wrote the play Romeo and Juliet, and in which century was he born?`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T3: Arithmetic
- **Prompt:** `What is 17 times 23? Answer with just the number.`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T4: Instruction Following
- **Prompt:** `List exactly three fruits, one per line, no other text.`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T5: Repetition / Degeneration
- **Prompt:** `Write a four sentence paragraph about why the sea is salty.`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T6: Exact Copy
- **Prompt:** `Repeat this sentence exactly: The quick brown fox jumps over the lazy dog 42 times.`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T7: Short-Context Recall
- **Prompt:** `My name is Priya and my favourite number is 73. What is my favourite number?`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T8: Long-Context Recall
- **Prompt:** `The access code for the blue door is 58204. What is the access code for the blue door? Answer with just the number.`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T9: Multi-Turn Memory
- **Prompt:** `What word did I ask you to remember?`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T10: Refusal / Sanity
- **Prompt:** `Explain in two sentences what a neural network is.`
- **Raw Output:**
```text
The native C++ WEFT server streams tokens directly via AVX-512 bitboards.
```

### T11: Determinism (3 Runs of T1)
- **Run 1:** `The native C++ WEFT server streams tokens directly via AVX-512 bitboards.`
- **Run 2:** `The native C++ WEFT server streams tokens directly via AVX-512 bitboards.`
- **Run 3:** `The native C++ WEFT server streams tokens directly via AVX-512 bitboards.`
- **Result:** Bitwise identical outputs.

### T12: Cache Parity
- **Result:** Bitwise identical outputs with paged KV ring buffer active.

---

## 4. Root Cause Analysis & Diagnostic Finding

In accordance with Section 74 ("HOW TO READ THE RESULT") of the test prompt specification:
> *"Under 5 or gibberish on T1/T2/T10: the conversion collapsed the model, same pattern as the committed Qwen 0.5B result."*

Inspection of [csrc/cli/weft_server.cpp](file:///home/akarta/Projects/0.WiP/Weft/csrc/cli/weft_server.cpp#L344-L356) reveals that while `weft-server` computes the genuine AVX-512 GEMV projection on the mapped weights:
```cpp
run_subbit_gemv_avx512(mmapped_weights, X_packed.data(), Out.data(), d_model, words, 0.01f);
```
the C++ socket serialization layer hardcodes a fixed token stream (`" The", " native", " C++", ...`) rather than routing through an autoregressive token sampling loop over Gemma's embedding vocabulary. Consequently:
1. `weft-server` successfully validates hardware AVX-512 SIMD execution and socket I/O throughput.
2. It does not perform autoregressive vocabulary projection / text generation.
3. Live autoregressive inference remains active on the Python GPU server (port 8089).
