#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace weft {

/**
 * Tropical Toric Degenerations for attention state piecewise-linear coordinate projection.
 */
class TropicalToricCompressor {
public:
    static int32_t tropical_valuation(float x, float base = 2.0f) {
        if (std::abs(x) < 1e-7f) return -128;
        return static_cast<int32_t>(std::floor(std::log2(std::abs(x))));
    }

    static void project_fan(const float* src, int8_t* dst, size_t dim) {
        for (size_t i = 0; i < dim; ++i) {
            int32_t v = tropical_valuation(src[i]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            dst[i] = static_cast<int8_t>(v);
        }
    }
};

} // namespace weft
