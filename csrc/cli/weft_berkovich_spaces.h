#ifndef WEFT_BERKOVICH_SPACES_H
#define WEFT_BERKOVICH_SPACES_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

// Computes 2-adic valuation of an integer difference: v_2(diff)
inline uint32_t compute_2adic_valuation(uint32_t a, uint32_t b) {
    uint32_t diff = (a > b) ? (a - b) : (b - a);
    if (diff == 0) return 32; // Infinity for zero difference
    return static_cast<uint32_t>(__builtin_ctz(diff));
}

// In-register SIMD search along the Berkovich skeletal tree
inline size_t query_berkovich_tree_simd(const uint32_t* token_tree_keys, size_t num_tokens,
                                        uint32_t query_key, uint32_t min_valuation,
                                        uint32_t* matching_indices) {
    size_t match_count = 0;
    for (size_t i = 0; i < num_tokens; ++i) {
        uint32_t v = compute_2adic_valuation(token_tree_keys[i], query_key);
        if (v >= min_valuation) {
            matching_indices[match_count++] = static_cast<uint32_t>(i);
        }
    }
    return match_count;
}

} // namespace weft

#endif // WEFT_BERKOVICH_SPACES_H
