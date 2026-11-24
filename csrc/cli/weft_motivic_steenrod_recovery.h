#pragma once
// weft_motivic_steenrod_recovery.h
// Milestone Paper 269: Asymmetric Motivic Cohomological Operations & Steenrod Powers
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace weft {

// Vectorized in-register bit-flip error recovery:
// out_clean[i] = corrupted_weights[i] ^ syndrome_error_masks[i]
inline void avx512_steenrod_syndrome_recovery(
    const uint32_t* __restrict__ corrupted_weights,
    const uint32_t* __restrict__ error_masks,
    uint32_t* __restrict__ out_clean,
    size_t count // number of 32-bit words
) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512i vw = _mm512_loadu_si512(reinterpret_cast<const void*>(corrupted_weights + i));
        __m512i ve = _mm512_loadu_si512(reinterpret_cast<const void*>(error_masks + i));
        __m512i v_clean = _mm512_xor_si512(vw, ve);
        _mm512_storeu_si512(reinterpret_cast<void*>(out_clean + i), v_clean);
    }

    for (; i < count; ++i) {
        out_clean[i] = corrupted_weights[i] ^ error_masks[i];
    }
}

} // namespace weft
