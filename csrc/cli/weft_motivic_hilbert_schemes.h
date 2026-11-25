#ifndef WEFT_MOTIVIC_HILBERT_SCHEMES_H
#define WEFT_MOTIVIC_HILBERT_SCHEMES_H

#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace weft {

// AVX-512 Asymmetric Motivic Cohomological Operations on Hilbert Schemes SEU Recovery Kernel
// Inspects and sanitizes ternary weights {-1, 0, +1} in-register against single-event bit-flips.
// Valid ternary int8 range is [-1, 1]. Any out-of-range value (caused by SEU bit corruption)
// is snapped back to the nearest valid ternary lattice boundary.
inline size_t hilbert_scheme_seu_sanitize_avx512(
    int8_t* __restrict__ ternary_weights,
    size_t dim)
{
    __m512i vmin = _mm512_set1_epi8(-1);
    __m512i vmax = _mm512_set1_epi8(1);
    size_t errors_corrected = 0;

    size_t i = 0;
    for (; i + 64 <= dim; i += 64) {
        __m512i vw = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ternary_weights + i));

        // Detect out-of-bounds: vw < -1 or vw > 1
        __mmask64 lt_mask = _mm512_cmplt_epi8_mask(vw, vmin);
        __mmask64 gt_mask = _mm512_cmpgt_epi8_mask(vw, vmax);
        __mmask64 err_mask = lt_mask | gt_mask;

        if (err_mask) {
            // Clamp into valid range [-1, 1]
            __m512i vclamped = _mm512_max_epi8(vmin, _mm512_min_epi8(vmax, vw));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(ternary_weights + i), vclamped);

            errors_corrected += _mm_popcnt_u64(err_mask);
        }
    }

    for (; i < dim; ++i) {
        if (ternary_weights[i] < -1) {
            ternary_weights[i] = -1;
            errors_corrected++;
        } else if (ternary_weights[i] > 1) {
            ternary_weights[i] = 1;
            errors_corrected++;
        }
    }

    return errors_corrected;
}

} // namespace weft

#endif // WEFT_MOTIVIC_HILBERT_SCHEMES_H
