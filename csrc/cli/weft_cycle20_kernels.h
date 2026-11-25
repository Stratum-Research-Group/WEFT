#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace weft {

class Cycle20Kernels {
public:
    static size_t orbit_dim(size_t ambient_dim, size_t cone_dim) {
        return (ambient_dim >= cone_dim) ? (ambient_dim - cone_dim) : 0;
    }

    static size_t total_cohomology_dim(size_t h0, size_t h1) {
        return h0 + h1;
    }
};

} // namespace weft
