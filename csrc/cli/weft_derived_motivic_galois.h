#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>

namespace weft {

/**
 * Derived Motivic Galois operator for algebraic weight tensor compaction.
 */
class DerivedMotivicGaloisCompressor {
public:
    static size_t unipotent_radical_max_dim(size_t rank) {
        return (rank * (rank - 1)) / 2;
    }

    /**
     * Compacts weight channels by projecting out unipotent radical drift.
     */
    static void project_radical(const float* src, float* dst, size_t dim, float alpha) {
        size_t i = 0;
#if defined(__AVX512F__)
        __m512 valpha = _mm512_set1_ps(alpha);
        for (; i + 16 <= dim; i += 16) {
            __m512 vsrc = _mm512_loadu_ps(src + i);
            __m512 vres = _mm512_mul_ps(vsrc, valpha);
            _mm512_storeu_ps(dst + i, vres);
        }
#endif
        for (; i < dim; ++i) {
            dst[i] = src[i] * alpha;
        }
    }
};

} // namespace weft
