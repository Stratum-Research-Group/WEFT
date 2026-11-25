#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "csrc/cli/weft_motivic_k3_involutions.h"

int main() {
    std::cout << "[TEST] Running test_motivic_k3_involutions (Paper 314)..." << std::endl;

    const size_t dim = 32;
    std::vector<float> input(dim, 2.0f);
    std::vector<float> masked(dim, 0.0f);
    std::vector<int8_t> ternary(dim, 1);

    uint32_t seed = 0x12345678;

    weft::K3SideChannelMasker::apply_reflection_permutation(input.data(), masked.data(), dim, seed);

    // Verify values have absolute value 2.0 but varying signs
    int sign_flips = 0;
    for (size_t i = 0; i < dim; ++i) {
        assert(std::fabs(std::fabs(masked[i]) - 2.0f) < 1e-5f);
        if (masked[i] < 0.0f) sign_flips++;
    }
    std::cout << "  Negative masked sign count: " << sign_flips << " / " << dim << std::endl;
    assert(sign_flips > 5 && sign_flips < 27); // Non-trivial pseudo-random distribution

    // Evaluating masked dot product must recover true unmasked sum: 32 * 2.0 = 64.0
    float unmasked_sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        unmasked_sum += input[i] * ternary[i];
    }
    float masked_dot = weft::K3SideChannelMasker::evaluate_masked_dot_product(masked.data(), ternary.data(), dim, seed);
    std::cout << "  Masked dot product recovered: " << masked_dot << " (expected " << unmasked_sum << ")" << std::endl;
    assert(std::fabs(masked_dot - unmasked_sum) < 1e-4f);

    std::cout << "[PASS] test_motivic_k3_involutions succeeded." << std::endl;
    return 0;
}
