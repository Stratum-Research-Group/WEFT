#include "csrc/cli/weft_derived_intersection_homology.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim = 128;
    std::vector<float> input(dim);
    std::vector<float> gamma(dim, 1.0f);
    std::vector<float> output(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        input[i] = (i % 2 == 0) ? 2.5f : -1.5f;
    }
    // Inject singular outlier (simulating stratum crossing)
    input[10] = 50.0f;

    weft::perverse_sheaf_normalize_avx512(input.data(), gamma.data(), output.data(), dim);

    float out_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        assert(!std::isnan(output[i]));
        assert(!std::isinf(output[i]));
        out_sq += output[i] * output[i];
    }

    // Check that singular outlier was tempered by perverse truncation
    assert(std::abs(output[10]) < 10.0f);
    assert(out_sq > 0.0f);

    std::cout << "[PASS] Derived Intersection Homology Normalization Unit Test passed.\n";
    return 0;
}
