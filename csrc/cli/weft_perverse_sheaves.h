#ifndef WEFT_PERVERSE_SHEAVES_H
#define WEFT_PERVERSE_SHEAVES_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct PerverseSupportState {
    int32_t stratum_dim;
    int32_t cohomology_deg;
    bool is_perverse;
};

// Checks middle perversity condition: stratum_dim + cohomology_deg <= 0
inline bool check_middle_perversity(int32_t stratum_dim, int32_t cohomology_deg) {
    return (stratum_dim + cohomology_deg <= 0);
}

// In-register SIMD projection preserving perverse intersection cohomology support
inline void project_perverse_stratum_gradients_simd(const float* raw_grads,
                                                    const int32_t* stratum_dims,
                                                    const int32_t* cohomology_degs,
                                                    size_t dim, float* perverse_grads) {
    for (size_t i = 0; i < dim; ++i) {
        if (check_middle_perversity(stratum_dims[i], cohomology_degs[i])) {
            perverse_grads[i] = raw_grads[i];
        } else {
            perverse_grads[i] = 0.0f; // Vanishing off-support perverse component
        }
    }
}

} // namespace weft

#endif // WEFT_PERVERSE_SHEAVES_H
