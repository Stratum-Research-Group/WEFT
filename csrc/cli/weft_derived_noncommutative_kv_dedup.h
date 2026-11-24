#pragma once
// weft_derived_noncommutative_kv_dedup.h
// Milestone Paper 243: Native AVX-512 Derived Non-Commutative KV Deduplication
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include <vector>

namespace weft {

// Compute cyclic trace for a vector of length N using AVX-512:
// Tr = sum_{i=0}^{N-1} a[i] * b[(i+1)%N]
inline float avx512_compute_cyclic_trace(
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t length
) {
    if (length == 0) return 0.0f;
    float trace_sum = 0.0f;

    // For cyclic multiplication: b_shifted[i] = b[(i+1)%N]
    // Vectorized main loop
    size_t i = 0;
    for (; i + 16 <= length; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        // Load next shifted values
        float temp_b[16];
        for (int k = 0; k < 16; ++k) {
            temp_b[k] = b[(i + k + 1) % length];
        }
        __m512 vb_shift = _mm512_loadu_ps(temp_b);
        __m512 prod = _mm512_mul_ps(va, vb_shift);
        trace_sum += _mm512_reduce_add_ps(prod);
    }

    for (; i < length; ++i) {
        trace_sum += a[i] * b[(i + 1) % length];
    }

    return trace_sum;
}

// Deduplicate candidate block against canonical reference:
// returns true if relative trace difference <= tolerance.
inline bool is_kv_block_duplicate(
    const float* __restrict__ cand_a,
    const float* __restrict__ cand_b,
    const float* __restrict__ canon_a,
    const float* __restrict__ canon_b,
    size_t length,
    float tolerance
) {
    float tr_cand = avx512_compute_cyclic_trace(cand_a, cand_b, length);
    float tr_canon = avx512_compute_cyclic_trace(canon_a, canon_b, length);
    return std::fabs(tr_cand - tr_canon) <= tolerance;
}

} // namespace weft
