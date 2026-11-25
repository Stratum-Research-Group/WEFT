#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace weft {

class MotivicIntersectionHomology {
public:
    static size_t middle_perversity(size_t k) {
        return (k >= 2) ? ((k - 2) / 2) : 0;
    }
};

} // namespace weft
