#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace weft {

/**
 * Motivic Picard-Lefschetz Monodromy for Multi-Tenant DRAM Sharding.
 * Applies involution s_delta(x) = x + <x, delta> * delta with <delta, delta> = -2.
 */
class PicardLefschetzMonodromy {
public:
    static void reflect(const float* x, const float* delta, float* out, size_t dim) {
        float inner_product = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            inner_product += x[i] * delta[i];
        }
        for (size_t i = 0; i < dim; ++i) {
            out[i] = x[i] + inner_product * delta[i];
        }
    }
};

} // namespace weft
