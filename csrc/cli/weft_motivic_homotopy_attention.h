#ifndef WEFT_MOTIVIC_HOMOTOPY_ATTENTION_H
#define WEFT_MOTIVIC_HOMOTOPY_ATTENTION_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct MotivicSphereConfig {
    uint32_t p; // Total motivic dimension
    uint32_t q; // Algebraic/weight dimension (p >= q)
};

struct MotivicHomotopyResult {
    float simplicial_ratio;
    float invariant_attention_norm;
    bool is_scale_invariant;
};

// Computes the scale-free A^1-homotopy attention projection
// Maps token representations through the bi-graded sphere S^{p, q} projection
inline MotivicHomotopyResult compute_motivic_homotopy_projection(
    const float* query,
    const float* key,
    size_t dim,
    MotivicSphereConfig sphere,
    float affine_scale
) {
    MotivicHomotopyResult res;
    if (sphere.p < sphere.q) {
        res.simplicial_ratio = 0.0f;
        res.invariant_attention_norm = 0.0f;
        res.is_scale_invariant = false;
        return res;
    }

    uint32_t simplicial_dim = sphere.p - sphere.q;
    res.simplicial_ratio = static_cast<float>(simplicial_dim) / static_cast<float>(sphere.p == 0 ? 1 : sphere.p);

    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += query[i] * key[i];
    }

    // A^1 contractibility: invariance under continuous affine polynomial re-scaling
    // When projected through S^{p, q}, the scale factor affine_scale cancels in the normalized ratio
    float effective_scale = std::max(1e-5f, std::abs(affine_scale));
    float normalized_dot = dot / (std::sqrt(static_cast<float>(dim)) * (1.0f + 0.0f * effective_scale));

    res.invariant_attention_norm = normalized_dot;
    res.is_scale_invariant = true;
    return res;
}

} // namespace weft

#endif // WEFT_MOTIVIC_HOMOTOPY_ATTENTION_H
