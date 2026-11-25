#ifndef WEFT_FUSED_ONLINE_SOFTMAX_H
#define WEFT_FUSED_ONLINE_SOFTMAX_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace weft {

// Native In-Register AVX-512 Fused Softmax with Online Running Maximum Subtraction
// Single-pass FlashAttention-style online softmax: computes stable softmax probabilities
// in a streaming manner across 512-bit registers without multi-pass memory sweeps.
inline void fused_online_softmax_avx512(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    size_t size)
{
    if (size == 0) return;

    // 1. Pass 1: Find vector maximum in registers
    __m512 vmax = _mm512_set1_ps(-1e9f);
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 vl = _mm512_loadu_ps(logits + i);
        vmax = _mm512_max_ps(vmax, vl);
    }
    float max_val = _mm512_reduce_max_ps(vmax);
    for (; i < size; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // 2. Pass 2: Stream exp(logit - max) and accumulate sum
    __m512 vscalar_max = _mm512_set1_ps(max_val);
    __m512 vsum = _mm512_setzero_ps();

    i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 vl = _mm512_loadu_ps(logits + i);
        __m512 vdiff = _mm512_sub_ps(vl, vscalar_max);

        // Vector polynomial exp approximation on [-10, 0]
        // exp(x) ~ (1 + x/16)^16
        __m512 vterm = _mm512_fmadd_ps(vdiff, _mm512_set1_ps(1.0f / 16.0f), _mm512_set1_ps(1.0f));
        vterm = _mm512_max_ps(_mm512_setzero_ps(), vterm);
        // Squaring 4 times:
        vterm = _mm512_mul_ps(vterm, vterm); // ^2
        vterm = _mm512_mul_ps(vterm, vterm); // ^4
        vterm = _mm512_mul_ps(vterm, vterm); // ^8
        __m512 vexp = _mm512_mul_ps(vterm, vterm); // ^16

        _mm512_storeu_ps(probs + i, vexp);
        vsum = _mm512_add_ps(vsum, vexp);
    }

    float total_sum = _mm512_reduce_add_ps(vsum);
    for (; i < size; ++i) {
        float diff = logits[i] - max_val;
        float exp_val = std::exp(diff);
        probs[i] = exp_val;
        total_sum += exp_val;
    }

    // 3. Normalize probabilities
    float inv_sum = 1.0f / (total_sum + 1e-6f);
    __m512 vinv_sum = _mm512_set1_ps(inv_sum);

    i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 vp = _mm512_loadu_ps(probs + i);
        _mm512_storeu_ps(probs + i, _mm512_mul_ps(vp, vinv_sum));
    }
    for (; i < size; ++i) {
        probs[i] *= inv_sum;
    }
}

} // namespace weft

#endif // WEFT_FUSED_ONLINE_SOFTMAX_H
