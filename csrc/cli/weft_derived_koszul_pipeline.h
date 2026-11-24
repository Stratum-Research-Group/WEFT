#pragma once
// weft_derived_koszul_pipeline.h
// Milestone Paper 252: Native AVX-512 Derived Koszul Pipeline Parallelism
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>

namespace weft {

// Vectorized dual-stage forward-backward interleaving step:
// out_stage_fwd[i] = act_fwd[i] + dt * grad_bwd[i]
// out_stage_bwd[i] = grad_bwd[i] - dt * act_fwd[i]
inline void avx512_koszul_interleave_stages(
    const float* __restrict__ act_fwd,
    const float* __restrict__ grad_bwd,
    float* __restrict__ out_stage_fwd,
    float* __restrict__ out_stage_bwd,
    size_t count,
    float dt
) {
    __m512 v_dt = _mm512_set1_ps(dt);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 vf = _mm512_loadu_ps(act_fwd + i);
        __m512 vb = _mm512_loadu_ps(grad_bwd + i);

        __m512 v_fwd_next = _mm512_fmadd_ps(v_dt, vb, vf);
        __m512 v_bwd_next = _mm512_fnmadd_ps(v_dt, vf, vb);

        _mm512_storeu_ps(out_stage_fwd + i, v_fwd_next);
        _mm512_storeu_ps(out_stage_bwd + i, v_bwd_next);
    }

    for (; i < count; ++i) {
        float f = act_fwd[i];
        float b = grad_bwd[i];
        out_stage_fwd[i] = f + dt * b;
        out_stage_bwd[i] = b - dt * f;
    }
}

} // namespace weft
