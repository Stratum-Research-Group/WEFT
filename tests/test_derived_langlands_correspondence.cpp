#include "csrc/cli/weft_derived_langlands_correspondence.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim_in = 64;   // Draft dimension
    constexpr size_t dim_out = 128; // Target dimension

    std::vector<float> act_draft(dim_in, 1.0f);
    std::vector<float> hecke_eigenvals(dim_out, 1.5f);
    std::vector<float> act_target(dim_out, 0.0f);

    weft::derived_langlands_align_avx512(
        act_draft.data(), hecke_eigenvals.data(), act_target.data(),
        dim_in, dim_out, 1.0f);

    float target_energy = 0.0f;
    for (size_t j = 0; j < dim_out; ++j) {
        assert(!std::isnan(act_target[j]));
        assert(!std::isinf(act_target[j]));
        target_energy += act_target[j] * act_target[j];
    }

    assert(target_energy > 0.0f);

    std::cout << "[PASS] Derived Langlands Correspondence Alignment Unit Test passed (Target energy: " 
              << target_energy << ").\n";
    return 0;
}
