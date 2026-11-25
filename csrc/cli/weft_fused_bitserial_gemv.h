#ifndef WEFT_FUSED_BITSERIAL_GEMV_H
#define WEFT_FUSED_BITSERIAL_GEMV_H

#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace weft {

// AVX-512 Fused Bit-Serial Dot Product Kernel with Fast Popcount Reduction
// Computes matrix-vector product between 2-bit encoded ternary weights and activations.
// Ternary weights {-1, 0, +1} are encoded as:
// Bit-plane 0 (pos_bits): 1 if weight == +1, else 0
// Bit-plane 1 (neg_bits): 1 if weight == -1, else 0
// Dot product = (popcount(act_mask & pos_bits) - popcount(act_mask & neg_bits)) * scale
inline float fused_bitserial_gemv_avx512(
    const uint64_t* __restrict__ pos_bitplane, // 64 weights per uint64_t
    const uint64_t* __restrict__ neg_bitplane,
    const uint64_t* __restrict__ act_bitplane,  // Quantized activation sign/magnitude mask
    size_t num_u64_blocks,
    float scale = 0.5f)
{
    int64_t total_pos = 0;
    int64_t total_neg = 0;

    for (size_t b = 0; b < num_u64_blocks; ++b) {
        uint64_t act = act_bitplane[b];
        uint64_t pos_match = pos_bitplane[b] & act;
        uint64_t neg_match = neg_bitplane[b] & act;

        // Hardware single-cycle popcount instruction (__builtin_popcountll)
        total_pos += _mm_popcnt_u64(pos_match);
        total_neg += _mm_popcnt_u64(neg_match);
    }

    return static_cast<float>(total_pos - total_neg) * scale;
}

} // namespace weft

#endif // WEFT_FUSED_BITSERIAL_GEMV_H
