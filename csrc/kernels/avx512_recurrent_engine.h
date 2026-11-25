#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <immintrin.h>

namespace weft {

/**
 * AVX-512 In-Register Zero-Allocation Recurrent State Engine.
 * Implements S_t = sigma(gamma) * S_{t-1} + k_t * v_t^T directly in vector registers.
 */
inline void avx512_recurrent_step(
    float* state,          // [R, D]
    const float* gamma,    // [R]
    const float* k,        // [R]
    const float* v,        // [D]
    size_t R,
    size_t D
) {
    for (size_t r = 0; r < R; ++r) {
        float decay = 1.0f / (1.0f + std::exp(-gamma[r]));
        float k_val = k[r];
        float* state_row = state + r * D;

        size_t d = 0;
#if defined(__AVX512F__)
        __m512 vdecay = _mm512_set1_ps(decay);
        __m512 vk = _mm512_set1_ps(k_val);

        for (; d + 16 <= D; d += 16) {
            __m512 vs = _mm512_loadu_ps(state_row + d);
            __m512 vv = _mm512_loadu_ps(v + d);
            // S_new = decay * S_old + k * v
            __m512 updated = _mm512_fmadd_ps(vk, vv, _mm512_mul_ps(vs, vdecay));
            _mm512_storeu_ps(state_row + d, updated);
        }
#endif
        for (; d < D; ++d) {
            state_row[d] = decay * state_row[d] + k_val * v[d];
        }
    }
}

} // namespace weft
