#pragma once
// weft_motivic_slice_quantization.h
// Milestone Paper 266: Asymmetric Motivic Spectra & Slice Filtration
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized motivic slice precision allocation:
// out_precision[i] = b_min + alpha * sensitivities[i]
inline void avx512_motivic_slice_allocation(
    const float* __restrict__ sensitivities,
    float* __restrict__ out_precision,
    size_t num_layers,
    float b_min,
    float alpha
) {
    __m512 v_bmin = _mm512_set1_ps(b_min);
    __m512 v_alpha = _mm512_set1_ps(alpha);

    size_t i = 0;
    for (; i + 16 <= num_layers; i += 16) {
        __m512 v_sens = _mm512_loadu_ps(sensitivities + i);
        __m512 v_alloc = _mm512_fmadd_ps(v_alpha, v_sens, v_bmin);
        _mm512_storeu_ps(out_precision + i, v_alloc);
    }

    for (; i < num_layers; ++i) {
        out_precision[i] = b_min + alpha * sensitivities[i];
    }
}

} // namespace weft
