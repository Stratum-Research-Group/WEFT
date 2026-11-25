#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace weft {

/**
 * Derived Motivic Hodge Structure for tensor quantization level stratification.
 */
class MotivicHodgeQuantizer {
public:
    static size_t truncation_error(size_t total_variance, size_t retained_variance) {
        return (total_variance > retained_variance) ? (total_variance - retained_variance) : 0;
    }
};

} // namespace weft
