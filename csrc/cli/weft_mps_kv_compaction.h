// weft_mps_kv_compaction.h
// Native AVX-512 Matrix Product State (MPS) Schmidt Contractions and Entanglement Pruning.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace weft {

// Compute von Neumann entanglement entropy: S = -sum lambda_i^2 * log(lambda_i^2)
inline float compute_entanglement_entropy(const float* lambdas, int chi) {
    float entropy = 0.0f;
    for (int i = 0; i < chi; ++i) {
        float p = lambdas[i] * lambdas[i];
        if (p > 1e-12f) {
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

// Truncate Schmidt singular values to maximum bond dimension chi_max and re-normalize trace
inline int truncate_and_renormalize_mps_avx512(const float* lambdas_in, int chi_in,
                                               float* lambdas_out, int chi_max) {
    int chi_out = std::min(chi_in, chi_max);

    // Copy top chi_out singular values
    for (int i = 0; i < chi_out; ++i) {
        lambdas_out[i] = lambdas_in[i];
    }

    // Compute squared sum using AVX-512
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= chi_out; i += 16) {
        __m512 v = _mm512_loadu_ps(&lambdas_out[i]);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    float norm_sq = _mm512_reduce_add_ps(acc);
    for (; i < chi_out; ++i) {
        norm_sq += lambdas_out[i] * lambdas_out[i];
    }

    // Re-normalize trace sum to 1.0
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    __m512 v_inv = _mm512_set1_ps(inv_norm);
    i = 0;
    for (; i + 16 <= chi_out; i += 16) {
        __m512 v = _mm512_loadu_ps(&lambdas_out[i]);
        v = _mm512_mul_ps(v, v_inv);
        _mm512_storeu_ps(&lambdas_out[i], v);
    }
    for (; i < chi_out; ++i) {
        lambdas_out[i] *= inv_norm;
    }

    return chi_out;
}

} // namespace weft
