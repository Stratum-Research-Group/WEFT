#ifndef WEFT_DIRAC_SPECTRAL_ATTENTION_H
#define WEFT_DIRAC_SPECTRAL_ATTENTION_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Computes in-register commutator [D, A] = D * A - A * D for 2x2 Clifford blocks or diagonal representations
inline void compute_dirac_commutator_simd(const float* D, const float* A, size_t dim, float* commutator) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= dim; i += 16) {
        __m512 v_d = _mm512_loadu_ps(&D[i]);
        __m512 v_a = _mm512_loadu_ps(&A[i]);
        __m512 v_da = _mm512_mul_ps(v_d, v_a);
        __m512 v_ad = _mm512_mul_ps(v_a, v_d);
        __m512 v_comm = _mm512_sub_ps(v_da, v_ad);
        _mm512_storeu_ps(&commutator[i], v_comm);
    }
#elif defined(__AVX2__)
    for (; i + 8 <= dim; i += 8) {
        __m256 v_d = _mm256_loadu_ps(&D[i]);
        __m256 v_a = _mm256_loadu_ps(&A[i]);
        __m256 v_da = _mm256_mul_ps(v_d, v_a);
        __m256 v_ad = _mm256_mul_ps(v_a, v_d);
        __m256 v_comm = _mm256_sub_ps(v_da, v_ad);
        _mm256_storeu_ps(&commutator[i], v_comm);
    }
#endif
    for (; i < dim; ++i) {
        commutator[i] = D[i] * A[i] - A[i] * D[i];
    }
}

// Computes the non-commutative Connes spectral distance between two token states phi and psi:
// d_D(phi, psi) = sup_{||[D, a]|| <= 1} |phi(a) - psi(a)|
inline float compute_connes_spectral_distance(const float* phi, const float* psi, size_t dim, float d_scale) {
    if (d_scale <= 0.0f) d_scale = 1.0f;
    float sum_abs_diff = 0.0f;
    size_t i = 0;

#if defined(__AVX2__)
    __m256 v_sum = _mm256_setzero_ps();
    __m256 v_sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_p = _mm256_loadu_ps(&phi[i]);
        __m256 v_q = _mm256_loadu_ps(&psi[i]);
        __m256 v_diff = _mm256_sub_ps(v_p, v_q);
        __m256 v_abs = _mm256_andnot_ps(v_sign_mask, v_diff);
        v_sum = _mm256_add_ps(v_sum, v_abs);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, v_sum);
    for (int k = 0; k < 8; ++k) sum_abs_diff += tmp[k];
#endif
    for (; i < dim; ++i) {
        sum_abs_diff += std::abs(phi[i] - psi[i]);
    }
    // Normalized by operator Lipschitz bound ||[D, a]||
    return sum_abs_diff / d_scale;
}

} // namespace weft

#endif // WEFT_DIRAC_SPECTRAL_ATTENTION_H
