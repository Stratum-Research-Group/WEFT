#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "../csrc/cli/weft_fused_rmsnorm.h"

int main() {
    std::cout << "[TEST] Running Weft Fused In-Register RMSNorm Verification..." << std::endl;

    const size_t dim = 64;
    std::vector<float> input(dim);
    std::vector<float> weight(dim, 1.0f);
    std::vector<float> output(dim, 0.0f);

    for (size_t i = 0; i < dim; ++i) {
        input[i] = (i % 2 == 0) ? 2.0f : -2.0f;
    }

    weft::fused_rmsnorm_avx512(input.data(), weight.data(), output.data(), dim);

    // RMS of {2, -2} is 2.0. So input / 2.0 * 1.0 = {1, -1}
    for (size_t i = 0; i < dim; ++i) {
        float expected = (i % 2 == 0) ? 1.0f : -1.0f;
        assert(std::abs(output[i] - expected) < 1e-3f);
    }

    std::cout << "[PASS] Fused RMSNorm normalized activations verified accurately." << std::endl;
    std::cout << "[ALL PASSED] Fused RMSNorm tests successful." << std::endl;
    return 0;
}
