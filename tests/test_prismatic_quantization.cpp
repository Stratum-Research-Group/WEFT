#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_prismatic_quantization.h"

int main() {
    std::cout << "[TEST] Running Prismatic Cohomology Quantization Invariant Test..." << std::endl;

    // Test 1: Eisenstein polynomial evaluation: u=2, e=2, p=3 -> 2^2 + 3 = 7
    uint32_t e_val = weft::eval_eisenstein_poly(2, 2, 3);
    std::cout << "  E(2) = 2^2 + 3: " << e_val << " (Expected: 7)" << std::endl;
    assert(e_val == 7);

    // Test 2: In-register Nygaard filtration: u=1, e=1, p=3 -> E(1) = 1 + 3 = 4
    // phi(w) = w^2. Divisible by 4 if w is even (w % 2 == 0)
    const size_t DIM = 16;
    std::vector<uint32_t> weights(DIM);
    std::vector<uint32_t> filtered(DIM);

    for (size_t i = 0; i < DIM; ++i) {
        weights[i] = static_cast<uint32_t>(i);
    }

    size_t count = weft::filter_nygaard_prismatic_weights_simd(
        weights.data(), DIM, 1, 1, 3, filtered.data()
    );

    std::cout << "  Nygaard filtered count: " << count << " / " << DIM << std::endl;
    // Even numbers: 0, 2, 4, 6, 8, 10, 12, 14 -> 8 elements
    assert(count == 8);

    for (size_t i = 0; i < count; ++i) {
        assert(filtered[i] % 2 == 0);
    }

    std::cout << "[PASS] All Prismatic Cohomology Quantization invariants verified." << std::endl;
    return 0;
}
