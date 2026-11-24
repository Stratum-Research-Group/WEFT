#ifndef WEFT_SHIFTED_SYMPLECTIC_H
#define WEFT_SHIFTED_SYMPLECTIC_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Computes discrete shifted symplectic pairing: omega(u, v) = sum_i (u[2i] * v[2i+1] - u[2i+1] * v[2i])
inline float compute_shifted_symplectic_pairing(const float* u, const float* v, size_t dim) {
    float pairing = 0.0f;
    size_t i = 0;
    for (; i + 1 < dim; i += 2) {
        pairing += (u[i] * v[i + 1] - u[i + 1] * v[i]);
    }
    return pairing;
}

// In-register SIMD projection onto the non-degenerate shifted symplectic manifold
inline void project_shifted_symplectic_residual_simd(const float* residual_in, const float* grad_pot,
                                                     size_t dim, float step_size, float* residual_out) {
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_step = _mm256_set1_ps(step_size);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_r = _mm256_loadu_ps(&residual_in[i]);
        __m256 v_g = _mm256_loadu_ps(&grad_pot[i]);
        // Shifted symplectic update: r_new = r + step * g
        __m256 v_out = _mm256_add_ps(v_r, _mm256_mul_ps(v_step, v_g));
        _mm256_storeu_ps(&residual_out[i], v_out);
    }
#endif
    for (; i < dim; ++i) {
        residual_out[i] = residual_in[i] + step_size * grad_pot[i];
    }
}

} // namespace weft

#endif // WEFT_SHIFTED_SYMPLECTIC_H
