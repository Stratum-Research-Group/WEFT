#include "csrc/cli/weft_motivic_quiver_grassmannians.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim = 64;
    constexpr size_t rank = 4;

    std::vector<float> lora_A(dim * rank, 0.5f);
    std::vector<float> lora_h(rank, 1.0f);
    std::vector<float> output_y(dim, 10.0f);

    weft::quiver_adapter_compose_avx512(
        lora_A.data(), lora_h.data(), output_y.data(), dim, rank, 2.0f);

    // Expected addition per coordinate:
    // rank * (0.5 * 1.0 * 2.0) = 4 * 1.0 = 4.0f
    // output_y = 10.0 + 4.0 = 14.0f
    for (size_t d = 0; d < dim; ++d) {
        assert(!std::isnan(output_y[d]));
        assert(std::abs(output_y[d] - 14.0f) < 1e-4f);
    }

    std::cout << "[PASS] Motivic Quiver Grassmannians Dynamic Adapter Composer Unit Test passed (Output coordinate 0: " 
              << output_y[0] << ").\n";
    return 0;
}
