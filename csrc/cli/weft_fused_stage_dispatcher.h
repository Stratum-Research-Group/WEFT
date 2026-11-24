#pragma once
// weft_fused_stage_dispatcher.h
// In-Register AVX-512 Fused Kernel Composition Pipeline (Item 4)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <cmath>
#include <immintrin.h>
#include "weft_derived_koszul_pipeline.h"
#include "weft_tropical_wronskian_pruning.h"
#include "weft_motivic_lfunctions_precision.h"

namespace weft {

// Fused single-pass L1-cache pipeline:
// Executes Koszul dual interleaving, tropical head difference evaluation,
// and motivic precision scaling in a single contiguous register sweep.
inline void avx512_fused_stage_dispatch(
    const float* __restrict__ act_fwd,
    const float* __restrict__ grad_bwd,
    const float* __restrict__ head_ref,
    const float* __restrict__ regulators,
    float* __restrict__ out_fwd,
    float* __restrict__ out_bwd,
    float* __restrict__ out_head_diff,
    float* __restrict__ out_precision,
    size_t count,
    float dt,
    float base_bits,
    float min_bits,
    float mean_regulator,
    float inv_beta
) {
    __m512 v_dt = _mm512_set1_ps(dt);
    __m512 v_base = _mm512_set1_ps(base_bits);
    __m512 v_min = _mm512_set1_ps(min_bits);
    __m512 v_mean = _mm512_set1_ps(mean_regulator);
    __m512 v_inv_beta = _mm512_set1_ps(inv_beta);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        // 1. Koszul interleaving
        __m512 vf = _mm512_loadu_ps(act_fwd + i);
        __m512 vb = _mm512_loadu_ps(grad_bwd + i);
        __m512 v_fwd_next = _mm512_fmadd_ps(v_dt, vb, vf);
        __m512 v_bwd_next = _mm512_fnmadd_ps(v_dt, vf, vb);
        _mm512_storeu_ps(out_fwd + i, v_fwd_next);
        _mm512_storeu_ps(out_bwd + i, v_bwd_next);

        // 2. Tropical head valuation
        __m512 v_ref = _mm512_loadu_ps(head_ref + i);
        __m512 v_diff = _mm512_abs_ps(_mm512_sub_ps(vf, v_ref));
        _mm512_storeu_ps(out_head_diff + i, v_diff);

        // 3. Motivic precision allocation
        __m512 v_reg = _mm512_loadu_ps(regulators + i);
        __m512 v_reg_diff = _mm512_sub_ps(v_reg, v_mean);
        __m512 v_prec = _mm512_max_ps(_mm512_fmadd_ps(v_inv_beta, v_reg_diff, v_base), v_min);
        _mm512_storeu_ps(out_precision + i, v_prec);
    }

    for (; i < count; ++i) {
        float f = act_fwd[i];
        float b = grad_bwd[i];
        out_fwd[i] = f + dt * b;
        out_bwd[i] = b - dt * f;
        out_head_diff[i] = std::fabs(f - head_ref[i]);
        float p = base_bits + inv_beta * (regulators[i] - mean_regulator);
        out_precision[i] = (p < min_bits) ? min_bits : p;
    }
}

} // namespace weft
