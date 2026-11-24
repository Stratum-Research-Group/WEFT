#pragma once
// weft_motivic_polylog_rope.h
// Milestone Paper 260: Asymmetric Motivic Polylogarithms & Zagier Conjectures
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized RoPE phase dequantization:
// out_phases[i] = quant_phases[i] - scale * regulator_corrections[i]
inline void avx512_motivic_rope_dequant(
    const float* __restrict__ quant_phases,
    const float* __restrict__ regulator_corrections,
    float* __restrict__ out_phases,
    size_t count,
    float scale
) {
    __m512 v_scale = _mm512_set1_ps(scale);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v_qp = _mm512_loadu_ps(quant_phases + i);
        __m512 v_rc = _mm512_loadu_ps(regulator_corrections + i);
        __m512 v_out = _mm512_fnmadd_ps(v_rc, v_scale, v_qp);
        _mm512_storeu_ps(out_phases + i, v_out);
    }

    for (; i < count; ++i) {
        out_phases[i] = quant_phases[i] - scale * regulator_corrections[i];
    }
}

} // namespace weft
