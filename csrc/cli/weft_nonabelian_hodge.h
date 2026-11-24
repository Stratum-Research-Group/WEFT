#ifndef WEFT_NONABELIAN_HODGE_H
#define WEFT_NONABELIAN_HODGE_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct HitchinSpectralState {
    float tr1;
    float tr2;
    float s2;
};

// Computes the Hitchin base invariant s2 = 0.5 * (Tr(Phi)^2 - Tr(Phi^2))
inline HitchinSpectralState compute_hitchin_invariants(const float* phi_diag, size_t dim) {
    HitchinSpectralState st;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float x = phi_diag[i];
        sum1 += x;
        sum2 += x * x;
    }
    st.tr1 = sum1;
    st.tr2 = sum2;
    st.s2 = 0.5f * (sum1 * sum1 - sum2);
    return st;
}

// In-register SIMD rescaling of multi-head attention weights to invariant Hitchin spectral target
inline void rescale_head_to_hitchin_base_simd(float* head_weights, size_t dim, float target_s2) {
    HitchinSpectralState st = compute_hitchin_invariants(head_weights, dim);
    if (std::abs(st.s2) < 1e-7f) return;
    float scale = std::sqrt(std::abs(target_s2 / st.s2));

    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_scale = _mm256_set1_ps(scale);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_w = _mm256_loadu_ps(&head_weights[i]);
        _mm256_storeu_ps(&head_weights[i], _mm256_mul_ps(v_w, v_scale));
    }
#endif
    for (; i < dim; ++i) {
        head_weights[i] *= scale;
    }
}

} // namespace weft

#endif // WEFT_NONABELIAN_HODGE_H
