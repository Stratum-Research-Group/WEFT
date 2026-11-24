#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_homological_mirror.h"

int main() {
    std::cout << "[TEST] Running Homological Mirror Symmetry Cross-Attention Test..." << std::endl;

    // Test 1: Zero phase difference gives zero area and unit weight
    weft::LagrangianIntersection li = weft::compute_lagrangian_intersection(1.5f, 1.5f);
    std::cout << "  Zero phase diff: Area=" << li.area << ", Weight=" << li.morphism_weight << std::endl;
    assert(li.area == 0.0f);
    assert(std::abs(li.morphism_weight - 1.0f) < 1e-6f);

    // Test 2: In-register vector cross-attention execution
    const size_t DIM = 32;
    std::vector<float> Q(DIM, 1.0f);
    std::vector<float> K(DIM, 1.0f);
    std::vector<float> V(DIM, 2.5f);
    std::vector<float> out(DIM, 0.0f);

    weft::apply_homological_mirror_cross_attention(Q.data(), K.data(), V.data(), DIM, out.data());
    for (size_t i = 0; i < DIM; ++i) {
        assert(std::abs(out[i] - 2.5f) < 1e-5f);
    }
    std::cout << "  Aligned Lagrangian intersection yields exact value preservation: 2.5." << std::endl;

    std::cout << "[PASS] All Homological Mirror Symmetry Cross-Attention invariants verified." << std::endl;
    return 0;
}
