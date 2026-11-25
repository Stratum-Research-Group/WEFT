#include "csrc/cli/weft_tropical_grassmannian_pruner.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    constexpr size_t dim = 128;
    std::vector<float> input(dim, 0.0f);
    std::vector<float> output(dim, 0.0f);

    // Set 40 channels above threshold (e.g. 0.2f), rest zero or small noise (0.01f)
    for (size_t i = 0; i < dim; ++i) {
        if (i < 40) {
            input[i] = 0.25f;
        } else {
            input[i] = 0.01f;
        }
    }

    size_t active = weft::tropical_grassmannian_prune_avx512(input.data(), output.data(), dim, 0.05f);

    assert(active == 40);
    for (size_t i = 0; i < 40; ++i) {
        assert(output[i] == 0.25f);
    }
    for (size_t i = 40; i < dim; ++i) {
        assert(output[i] == 0.0f);
    }

    std::cout << "[PASS] Tropical Grassmannian Sparsification Unit Test passed (Active: " 
              << active << " / " << dim << ").\n";
    return 0;
}
