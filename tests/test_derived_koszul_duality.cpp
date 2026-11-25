#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "csrc/cli/weft_derived_koszul_duality.h"

int main() {
    std::cout << "[TEST] Running test_derived_koszul_duality (Paper 312)..." << std::endl;

    const size_t dim = 32;
    std::vector<float> mod_a(dim, 1.0f);
    std::vector<float> mod_b(dim, 3.0f);
    std::vector<float> fused(dim, 0.0f);
    std::vector<int8_t> ternary(dim, 1);

    // alpha = 0.5: fused[i] = 1.0 * 0.5 + 3.0 * 0.5 = 2.0
    weft::OperadicKoszulComposer::compose_binary_tree(mod_a.data(), mod_b.data(), fused.data(), dim, 0.5f);

    for (size_t i = 0; i < dim; ++i) {
        assert(std::fabs(fused[i] - 2.0f) < 1e-5f);
    }
    std::cout << "  Fused vector value: " << fused[0] << " (expected 2.0)" << std::endl;

    float score = weft::OperadicKoszulComposer::evaluate_operad_norm(fused.data(), ternary.data(), dim);
    std::cout << "  Operad projection score: " << score << " (expected 64.0)" << std::endl;
    assert(std::fabs(score - 64.0f) < 1e-4f);

    std::cout << "[PASS] test_derived_koszul_duality succeeded." << std::endl;
    return 0;
}
