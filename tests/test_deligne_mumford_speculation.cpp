#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_deligne_mumford_speculation.h"

int main() {
    std::cout << "[TEST] Running Deligne-Mumford Moduli Stack Speculative Routing Test..." << std::endl;

    // Test 1: Harmonic cross-ratio calculation: x1=0, x2=inf (approximated), x3=1, x4=-1
    // For x1=0, x2=2, x3=1, x4=3:
    // num = (0 - 1) * (2 - 3) = (-1)*(-1) = 1
    // den = (0 - 3) * (2 - 1) = (-3)*(1) = -3
    // lambda = -1/3 = -0.333333
    float lam = weft::compute_projective_cross_ratio(0.0f, 2.0f, 1.0f, 3.0f);
    std::cout << "  Computed cross-ratio: " << lam << std::endl;
    assert(std::abs(lam - (-1.0f / 3.0f)) < 1e-5f);

    // Test 2: Stability evaluation: degenerate when x1 == x3 (lambda = 0)
    bool stable_clean = weft::is_speculative_branch_stable(0.0f, 2.0f, 1.0f, 3.0f, 0.1f);
    bool stable_degen = weft::is_speculative_branch_stable(1.0f, 2.0f, 1.0f, 3.0f, 0.1f);
    std::cout << "  Stable clean: " << stable_clean << ", Degenerate: " << stable_degen << std::endl;
    assert(stable_clean == true);
    assert(stable_degen == false);

    // Test 3: Batch SIMD filter
    std::vector<float> x1 = {0.0f, 1.0f, 5.0f};
    std::vector<float> x2 = {2.0f, 2.0f, 10.0f};
    std::vector<float> x3 = {1.0f, 1.0f, 7.0f};
    std::vector<float> x4 = {3.0f, 3.0f, 12.0f};
    std::vector<uint32_t> stable_idx(3);

    size_t num_stable = weft::filter_stable_speculative_branches_simd(
        x1.data(), x2.data(), x3.data(), x4.data(), 3, 0.1f, stable_idx.data()
    );

    std::cout << "  Retained stable branches: " << num_stable << " / 3" << std::endl;
    assert(num_stable == 2);
    assert(stable_idx[0] == 0);
    assert(stable_idx[1] == 2);

    std::cout << "[PASS] All Deligne-Mumford Speculative Routing invariants verified." << std::endl;
    return 0;
}
