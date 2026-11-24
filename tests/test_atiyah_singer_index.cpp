#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_atiyah_singer_index.h"

int main() {
    std::cout << "[TEST] Running Atiyah-Singer Index Theory & Topological Obstruction Test..." << std::endl;

    // Test 1: Analytical index calculation
    weft::DiracIndexState idx = weft::compute_dirac_analytical_index(4, 1);
    std::cout << "  Analytical index: " << idx.analytical_index << std::endl;
    assert(idx.analytical_index == 3);

    // Test 2: In-register Chern c1 class evaluation
    const size_t DIM = 32;
    std::vector<float> f01(DIM, 1.5707963268f);
    std::vector<float> f10(DIM, 0.0f);
    float c1 = weft::compute_chern_c1_simd(f01.data(), f10.data(), DIM);
    std::cout << "  Computed Chern c1: " << c1 << std::endl;
    assert(c1 > 0.0f);

    // Test 3: Topological anchor gradient injection
    std::vector<float> grads(DIM, 0.0f);
    weft::enforce_topological_anchor(grads.data(), DIM, idx.analytical_index, 0.25f);
    for (size_t i = 0; i < DIM; ++i) {
        assert(std::abs(grads[i] - 0.25f) < 1e-6f);
    }

    std::cout << "[PASS] All Atiyah-Singer Index Theory invariants verified." << std::endl;
    return 0;
}
