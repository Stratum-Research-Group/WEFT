#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace weft {

class TropicalToricCompactifier {
public:
    static float clamp_coordinate(float val, float bound) {
        return std::min(val, bound);
    }
};

} // namespace weft
