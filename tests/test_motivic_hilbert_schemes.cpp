#include "csrc/cli/weft_motivic_hilbert_schemes.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    constexpr size_t dim = 128;
    std::vector<int8_t> weights(dim, 0);

    // Set valid weights
    for (size_t i = 0; i < dim; ++i) {
        weights[i] = static_cast<int8_t>((i % 3) - 1); // -1, 0, +1
    }

    // Inject 3 SEU bit-flips (invalid numbers)
    weights[10] = 5;
    weights[42] = -12;
    weights[99] = 100;

    size_t corrected = weft::hilbert_scheme_seu_sanitize_avx512(weights.data(), dim);

    assert(corrected == 3);
    assert(weights[10] == 1);
    assert(weights[42] == -1);
    assert(weights[99] == 1);

    std::cout << "[PASS] Motivic Hilbert Schemes SEU Recovery Unit Test passed (Corrected " 
              << corrected << " errors).\n";
    return 0;
}
