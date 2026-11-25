#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace weft {

class MotivicGaloisDeformations {
public:
    static size_t max_obstruction_bound(size_t d) {
        return (d * (d - 1)) / 2;
    }
};

} // namespace weft
