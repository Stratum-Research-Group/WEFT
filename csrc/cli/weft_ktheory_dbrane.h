#ifndef WEFT_KTHEORY_DBRANE_H
#define WEFT_KTHEORY_DBRANE_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct KTheoryChargeState {
    int32_t charge_q;
    float phase;
};

// Computes the integer topological charge winding: Q = round(phase / (2 * pi))
inline int32_t compute_topological_charge(float phase) {
    constexpr float TWO_PI = 6.283185307179586f;
    return static_cast<int32_t>(std::round(phase / TWO_PI));
}

// In-register SIMD projection onto the discrete topological charge lattice
inline void snap_to_dbrane_charge_lattice_simd(const float* phases, size_t dim, float* anchored_phases) {
    constexpr float TWO_PI = 6.283185307179586f;
    constexpr float INV_TWO_PI = 1.0f / TWO_PI;
    size_t i = 0;
#if defined(__AVX2__)
    __m256 v_two_pi = _mm256_set1_ps(TWO_PI);
    __m256 v_inv_two_pi = _mm256_set1_ps(INV_TWO_PI);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_p = _mm256_loadu_ps(&phases[i]);
        __m256 v_div = _mm256_mul_ps(v_p, v_inv_two_pi);
        __m256 v_round = _mm256_round_ps(v_div, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 v_out = _mm256_mul_ps(v_round, v_two_pi);
        _mm256_storeu_ps(&anchored_phases[i], v_out);
    }
#endif
    for (; i < dim; ++i) {
        float q = std::round(phases[i] * INV_TWO_PI);
        anchored_phases[i] = q * TWO_PI;
    }
}

} // namespace weft

#endif // WEFT_KTHEORY_DBRANE_H
