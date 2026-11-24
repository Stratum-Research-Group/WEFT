#ifndef WEFT_CLUSTER_ALGEBRA_ROUTING_H
#define WEFT_CLUSTER_ALGEBRA_ROUTING_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Evaluates a single Ptolemy cluster mutation exchange: x_k' = (x_i * x_l + x_j * x_m) / x_k
inline float mutate_cluster_variable(float xi, float xl, float xj, float xm, float xk) {
    if (std::abs(xk) < 1e-7f) xk = 1e-7f; // Guard against division by zero
    float num = xi * xl + xj * xm;
    return num / xk;
}

// In-register SIMD execution of cluster seed mutation across attention head parameters
inline void mutate_attention_cluster_seed_simd(const float* xi, const float* xl,
                                               const float* xj, const float* xm,
                                               const float* xk, size_t dim,
                                               float* xk_mutated) {
    size_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= dim; i += 8) {
        __m256 v_xi = _mm256_loadu_ps(&xi[i]);
        __m256 v_xl = _mm256_loadu_ps(&xl[i]);
        __m256 v_xj = _mm256_loadu_ps(&xj[i]);
        __m256 v_xm = _mm256_loadu_ps(&xm[i]);
        __m256 v_xk = _mm256_loadu_ps(&xk[i]);

        __m256 num = _mm256_add_ps(_mm256_mul_ps(v_xi, v_xl), _mm256_mul_ps(v_xj, v_xm));
        __m256 v_out = _mm256_div_ps(num, v_xk);
        _mm256_storeu_ps(&xk_mutated[i], v_out);
    }
#endif
    for (; i < dim; ++i) {
        xk_mutated[i] = mutate_cluster_variable(xi[i], xl[i], xj[i], xm[i], xk[i]);
    }
}

} // namespace weft

#endif // WEFT_CLUSTER_ALGEBRA_ROUTING_H
