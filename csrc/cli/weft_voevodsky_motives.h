#ifndef WEFT_VOEVODSKY_MOTIVES_H
#define WEFT_VOEVODSKY_MOTIVES_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct MotivicTateCoefficients {
    float c_tate0;
    float c_tate1;
};

// Evaluates in-register projection of sub-bit weight onto Tate motivic invariant sub-manifold
inline void project_voevodsky_motives_simd(const float* weights_in, size_t dim,
                                           float c0, float c1, float* weights_out) {
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_c0 = _mm256_set1_ps(c0);
    __m256 v_c1 = _mm256_set1_ps(c1);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_w = _mm256_loadu_ps(&weights_in[i]);
        // Tate polynomial projection: w_proj = c0 * w + c1 * w^3 (normalized)
        __m256 v_w3 = _mm256_mul_ps(_mm256_mul_ps(v_w, v_w), v_w);
        __m256 v_out = _mm256_add_ps(_mm256_mul_ps(v_w, v_c0), _mm256_mul_ps(v_w3, v_c1));
        _mm256_storeu_ps(&weights_out[i], v_out);
    }
#endif
    for (; i < dim; ++i) {
        float w = weights_in[i];
        weights_out[i] = c0 * w + c1 * (w * w * w);
    }
}

} // namespace weft

#endif // WEFT_VOEVODSKY_MOTIVES_H
