#include "csrc/cli/weft_motivic_abelian_varieties.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t size = 256;
    constexpr float lattice_step = 0.666667f;
    std::vector<float> data(size);

    // Populate data with mean around 2 * lattice_step (~1.3333f)
    for (size_t i = 0; i < size; ++i) {
        data[i] = 1.3333f + ((i % 2 == 0) ? 0.1f : -0.1f);
    }

    weft::abelian_variety_recenter_avx512(data.data(), size, lattice_step);

    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    float new_mean = sum / static_cast<float>(size);

    // New mean should be close to 0.0f
    assert(std::abs(new_mean) < 0.05f);

    std::cout << "[PASS] Motivic Abelian Varieties Re-Centering Unit Test passed (New mean: " 
              << new_mean << ").\n";
    return 0;
}
