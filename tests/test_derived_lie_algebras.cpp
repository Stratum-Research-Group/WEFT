#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "csrc/cli/weft_derived_lie_algebras.h"

int main() {
    std::cout << "[TEST] Running test_derived_lie_algebras (Paper 309)..." << std::endl;

    const size_t dim = 32;
    std::vector<float> alpha(dim * dim, 0.0f);
    std::vector<float> d_alpha(dim * dim, 0.0f);
    std::vector<int8_t> base_ternary(dim * dim, 0);
    std::vector<float> edited(dim * dim, 0.0f);

    // Initialize symmetric or flat deformation field where [alpha, alpha] = 0
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            alpha[i * dim + j] = 0.001f * (i == j ? 1.0f : 0.0f);
            base_ternary[i * dim + j] = (i + j) % 3 - 1;
        }
    }

    float res = weft::MaurerCartanEditor::compute_mc_residual(alpha.data(), d_alpha.data(), dim);
    std::cout << "  Maurer-Cartan residual for abelian deformation: " << res << std::endl;
    assert(res < 1e-5f);

    bool ok = weft::MaurerCartanEditor::apply_integrable_edit(base_ternary.data(), alpha.data(), edited.data(), dim * dim);
    assert(ok);
    assert(std::fabs(edited[0] - (base_ternary[0] + alpha[0])) < 1e-5f);

    std::cout << "[PASS] test_derived_lie_algebras succeeded." << std::endl;
    return 0;
}
