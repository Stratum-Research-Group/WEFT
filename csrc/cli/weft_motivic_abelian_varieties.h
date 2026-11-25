#ifndef WEFT_MOTIVIC_ABELIAN_VARIETIES_H
#define WEFT_MOTIVIC_ABELIAN_VARIETIES_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>

namespace weft {

// AVX-512 Asymmetric Motivic Cohomological Operations on Abelian Varieties Re-Centering Kernel
// Projects accumulated layer biases onto the discrete period lattice step, removing secular drift.
inline void abelian_variety_recenter_avx512(
    float* __restrict__ data,
    size_t size,
    float lattice_step = 0.666667f) // Canonical Delta = 2/3 for ternary states
{
    if (size == 0) return;

    // 1. Compute global mean across 512-bit vector registers
    __m512 vsum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(data + i);
        vsum = _mm512_add_ps(vsum, v);
    }
    float sum = _mm512_reduce_add_ps(vsum);
    for (; i < size; ++i) {
        sum += data[i];
    }

    float mean = sum / static_cast<float>(size);

    // 2. Compute nearest lattice period coset
    float lattice_periods = std::round(mean / lattice_step);
    float offset = lattice_periods * lattice_step;

    // Only apply if there is an integral period offset
    if (std::abs(offset) < 1e-6f) return;

    __m512 voffset = _mm512_set1_ps(offset);

    // 3. Vectorized translation by period element
    i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(data + i);
        __m512 vcentered = _mm512_sub_ps(v, voffset);
        _mm512_storeu_ps(data + i, vcentered);
    }
    for (; i < size; ++i) {
        data[i] -= offset;
    }
}

} // namespace weft

#endif // WEFT_MOTIVIC_ABELIAN_VARIETIES_H
