#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_cluster_algebra_routing.h"

int main() {
    std::cout << "[TEST] Running Cluster Algebras & Grassmannian Seed Mutation Test..." << std::endl;

    // Test 1: Ptolemy exchange: xi=1, xl=2, xj=3, xm=4, xk=2
    // num = 1 * 2 + 3 * 4 = 2 + 12 = 14
    // xk' = 14 / 2 = 7
    float mutated = weft::mutate_cluster_variable(1.0f, 2.0f, 3.0f, 4.0f, 2.0f);
    std::cout << "  Scalar mutated cluster variable: " << mutated << " (Expected: 7.0)" << std::endl;
    assert(std::abs(mutated - 7.0f) < 1e-5f);

    // Test 2: In-register SIMD cluster mutation
    const size_t DIM = 32;
    std::vector<float> xi(DIM, 2.0f);
    std::vector<float> xl(DIM, 3.0f);
    std::vector<float> xj(DIM, 1.0f);
    std::vector<float> xm(DIM, 4.0f);
    std::vector<float> xk(DIM, 5.0f);
    std::vector<float> xk_out(DIM, 0.0f);

    // num = 2 * 3 + 1 * 4 = 6 + 4 = 10
    // xk' = 10 / 5 = 2.0
    weft::mutate_attention_cluster_seed_simd(
        xi.data(), xl.data(), xj.data(), xm.data(), xk.data(), DIM, xk_out.data()
    );

    for (size_t i = 0; i < DIM; ++i) {
        assert(std::abs(xk_out[i] - 2.0f) < 1e-5f);
    }
    std::cout << "  SIMD mutated seed variable: " << xk_out[0] << " (Expected: 2.0)" << std::endl;

    std::cout << "[PASS] All Cluster Algebra Attention Routing invariants verified." << std::endl;
    return 0;
}
