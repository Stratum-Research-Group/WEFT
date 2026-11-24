#ifndef WEFT_DESSINS_ENFANTS_H
#define WEFT_DESSINS_ENFANTS_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct BipartiteDessinEdge {
    uint32_t black_vertex;
    uint32_t white_vertex;
    float weight;
};

// Computes the Euler characteristic of a dessin ribbon graph: chi = V_b + V_w + F - E
inline int32_t compute_dessin_euler_characteristic(uint32_t vb, uint32_t vw, uint32_t faces, uint32_t edges) {
    return static_cast<int32_t>(vb + vw + faces) - static_cast<int32_t>(edges);
}

// In-register SIMD permutation routing of token features via dessin permutation sigma_0
inline void permute_tokens_by_dessin_simd(const float* tokens_in, const uint32_t* sigma_perm,
                                          size_t num_tokens, size_t dim, float* tokens_out) {
    for (size_t t = 0; t < num_tokens; ++t) {
        uint32_t src_idx = sigma_perm[t];
        const float* src = &tokens_in[src_idx * dim];
        float* dst = &tokens_out[t * dim];
        size_t d = 0;
#if defined(__AVX2__)
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(&src[d]);
            _mm256_storeu_ps(&dst[d], v);
        }
#endif
        for (; d < dim; ++d) {
            dst[d] = src[d];
        }
    }
}

} // namespace weft

#endif // WEFT_DESSINS_ENFANTS_H
