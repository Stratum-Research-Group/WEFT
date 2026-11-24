#ifndef WEFT_TROPICAL_TORIC_H
#define WEFT_TROPICAL_TORIC_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct TropicalAmoebaTerm {
    float alpha;
    float bias;
};

// Evaluates the piecewise-linear tropical Ronkin envelope for a single scalar
inline float eval_tropical_ronkin(float x, const TropicalAmoebaTerm* terms, size_t n_terms) {
    if (n_terms == 0) return 0.0f;
    float min_val = terms[0].alpha * x + terms[0].bias;
    for (size_t k = 1; k < n_terms; ++k) {
        float val = terms[k].alpha * x + terms[k].bias;
        if (val < min_val) {
            min_val = val;
        }
    }
    return min_val;
}

// In-register SIMD evaluation of tropical attention envelope across a vector of logits
inline void eval_tropical_amoeba_attention(const float* logits, size_t n_logits,
                                           const TropicalAmoebaTerm* terms, size_t n_terms,
                                           float* output) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n_logits; i += 16) {
        __m512 v_x = _mm512_loadu_ps(&logits[i]);
        __m512 v_min = _mm512_set1_ps(1e30f);
        for (size_t k = 0; k < n_terms; ++k) {
            __m512 v_alpha = _mm512_set1_ps(terms[k].alpha);
            __m512 v_bias = _mm512_set1_ps(terms[k].bias);
            __m512 v_val = _mm512_fmadd_ps(v_x, v_alpha, v_bias);
            v_min = _mm512_min_ps(v_min, v_val);
        }
        _mm512_storeu_ps(&output[i], v_min);
    }
#elif defined(__AVX2__)
    for (; i + 8 <= n_logits; i += 8) {
        __m256 v_x = _mm256_loadu_ps(&logits[i]);
        __m256 v_min = _mm256_set1_ps(1e30f);
        for (size_t k = 0; k < n_terms; ++k) {
            __m256 v_alpha = _mm256_set1_ps(terms[k].alpha);
            __m256 v_bias = _mm256_set1_ps(terms[k].bias);
            __m256 v_val = _mm256_fmadd_ps(v_x, v_alpha, v_bias);
            v_min = _mm256_min_ps(v_min, v_val);
        }
        _mm256_storeu_ps(&output[i], v_min);
    }
#endif
    for (; i < n_logits; ++i) {
        output[i] = eval_tropical_ronkin(logits[i], terms, n_terms);
    }
}

} // namespace weft

#endif // WEFT_TROPICAL_TORIC_H
