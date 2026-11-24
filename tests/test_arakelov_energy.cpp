#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_arakelov_energy.h"

int main() {
    std::cout << "[TEST] Running Arakelov Geometry Energy Minimization Invariant Test..." << std::endl;

    // Test 1: Height computation for zero vector: arch = log(1+0) = 0, non-arch = 0 -> total = 0
    const size_t DIM = 16;
    std::vector<float> zero_vec(DIM, 0.0f);
    weft::ArakelovHeightProfile h0 = weft::compute_arakelov_height_simd(zero_vec.data(), DIM);
    std::cout << "  Zero vector height: " << h0.total_height << std::endl;
    assert(h0.total_height == 0.0f);

    // Test 2: Height computation for non-zero vector
    std::vector<float> vec(DIM, 1.0f);
    weft::ArakelovHeightProfile h1 = weft::compute_arakelov_height_simd(vec.data(), DIM);
    std::cout << "  Vector [1...1] height: " << h1.total_height << " (Arch=" << h1.arch_height << ")" << std::endl;
    assert(h1.total_height > 0.0f);
    assert(h1.arch_height > 0.0f);

    // Test 3: Monotonic energy estimation
    float e0 = weft::estimate_joules_per_token(h0.total_height, 2.0f, 1.0f);
    float e1 = weft::estimate_joules_per_token(h1.total_height, 2.0f, 1.0f);
    std::cout << "  Estimated energy: E(h0)=" << e0 << " mJ, E(h1)=" << e1 << " mJ" << std::endl;
    assert(e0 < e1);

    std::cout << "[PASS] All Arakelov Geometry Energy Minimization invariants verified." << std::endl;
    return 0;
}
