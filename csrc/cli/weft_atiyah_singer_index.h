#ifndef WEFT_ATIYAH_SINGER_INDEX_H
#define WEFT_ATIYAH_SINGER_INDEX_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct DiracIndexState {
    int32_t dim_ker_plus;
    int32_t dim_ker_minus;
    int32_t analytical_index;
};

// Computes the analytical index: ind(D) = dim(ker D^+) - dim(ker D^-)
inline DiracIndexState compute_dirac_analytical_index(int32_t ker_plus, int32_t ker_minus) {
    DiracIndexState s;
    s.dim_ker_plus = ker_plus;
    s.dim_ker_minus = ker_minus;
    s.analytical_index = ker_plus - ker_minus;
    return s;
}

// In-register evaluation of Chern class c_1 from antisymmetric curvature tensor F
inline float compute_chern_c1_simd(const float* curvature_f01, const float* curvature_f10, size_t dim) {
    float c1_sum = 0.0f;
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_sum = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 v_01 = _mm256_loadu_ps(&curvature_f01[i]);
        __m256 v_10 = _mm256_loadu_ps(&curvature_f10[i]);
        __m256 v_diff = _mm256_sub_ps(v_01, v_10);
        v_sum = _mm256_add_ps(v_sum, v_diff);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, v_sum);
    for (int k = 0; k < 8; ++k) c1_sum += tmp[k];
#endif
    for (; i < dim; ++i) {
        c1_sum += (curvature_f01[i] - curvature_f10[i]);
    }
    return c1_sum * (0.5f / 3.1415926535f);
}

// Topological obstruction enforcement: if ind != 0, inject topological zero-mode anchor
inline void enforce_topological_anchor(float* gradients, size_t dim, int32_t index_val, float anchor_strength) {
    if (index_val == 0) return;
    float sign = (index_val > 0) ? 1.0f : -1.0f;
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_boost = _mm256_set1_ps(sign * anchor_strength);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_g = _mm256_loadu_ps(&gradients[i]);
        __m256 v_out = _mm256_add_ps(v_g, v_boost);
        _mm256_storeu_ps(&gradients[i], v_out);
    }
#endif
    for (; i < dim; ++i) {
        gradients[i] += sign * anchor_strength;
    }
}

} // namespace weft

#endif // WEFT_ATIYAH_SINGER_INDEX_H
