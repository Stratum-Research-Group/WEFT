#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace weft {

/**
 * Tropical Hyperplane Arrangements for activation sparsity bounds.
 */
class TropicalHyperplanesEngine {
public:
    static float tropical_hyperplane_eval(const float* x, const float* c, size_t dim) {
        float max_val = -1e9f;
        for (size_t i = 0; i < dim; ++i) {
            float val = x[i] + c[i];
            if (val > max_val) max_val = val;
        }
        return max_val;
    }
};

} // namespace weft
