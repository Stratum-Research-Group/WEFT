#pragma once
// weft_subbit_speculative_verifier.h
// Sub-Bit Dynamic Speculative Decoding Draft Model Verifier (Item 5)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized cosine-similarity verification between target logits and draft candidate logits:
// returns acceptance mask where similarity >= threshold
inline size_t avx512_verify_speculative_candidates(
    const float* __restrict__ target_logits,
    const float* __restrict__ draft_logits,
    float* __restrict__ out_residuals,
    size_t num_candidates,
    size_t vocab_dim,
    float threshold
) {
    size_t accepted_count = 0;

    for (size_t c = 0; c < num_candidates; ++c) {
        const float* t_ptr = target_logits + c * vocab_dim;
        const float* d_ptr = draft_logits + c * vocab_dim;

        __m512 v_dot = _mm512_setzero_ps();
        __m512 v_norm_t = _mm512_setzero_ps();
        __m512 v_norm_d = _mm512_setzero_ps();

        size_t v = 0;
        for (; v + 16 <= vocab_dim; v += 16) {
            __m512 vt = _mm512_loadu_ps(t_ptr + v);
            __m512 vd = _mm512_loadu_ps(d_ptr + v);

            v_dot = _mm512_fmadd_ps(vt, vd, v_dot);
            v_norm_t = _mm512_fmadd_ps(vt, vt, v_norm_t);
            v_norm_d = _mm512_fmadd_ps(vd, vd, v_norm_d);
        }

        float dot = _mm512_reduce_add_ps(v_dot);
        float norm_t = _mm512_reduce_add_ps(v_norm_t);
        float norm_d = _mm512_reduce_add_ps(v_norm_d);

        for (; v < vocab_dim; ++v) {
            dot += t_ptr[v] * d_ptr[v];
            norm_t += t_ptr[v] * t_ptr[v];
            norm_d += d_ptr[v] * d_ptr[v];
        }

        float denom = std::sqrt(norm_t * norm_d);
        float sim = (denom > 1e-8f) ? (dot / denom) : 0.0f;
        out_residuals[c] = sim;

        if (sim >= threshold) {
            accepted_count++;
        } else {
            break; // Speculative prefix rejection
        }
    }

    return accepted_count;
}

} // namespace weft
