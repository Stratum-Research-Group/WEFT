#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_dessins_enfants.h"

int main() {
    std::cout << "[TEST] Running Grothendieck Dessins d'Enfants Attention Test..." << std::endl;

    // Test 1: Euler characteristic on sphere: V_b=2, V_w=2, F=2, E=4 -> chi = 6 - 4 = 2
    int32_t chi = weft::compute_dessin_euler_characteristic(2, 2, 2, 4);
    std::cout << "  Computed Euler characteristic: " << chi << " (Expected: 2)" << std::endl;
    assert(chi == 2);

    // Test 2: In-register token permutation routing
    const size_t NUM_TOKENS = 4;
    const size_t DIM = 16;
    std::vector<float> tokens_in(NUM_TOKENS * DIM);
    std::vector<float> tokens_out(NUM_TOKENS * DIM, 0.0f);

    for (size_t t = 0; t < NUM_TOKENS; ++t) {
        for (size_t d = 0; d < DIM; ++d) {
            tokens_in[t * DIM + d] = static_cast<float>(t + 1);
        }
    }

    // Cyclic permutation sigma = (0 1 2 3) -> 1, 2, 3, 0
    std::vector<uint32_t> sigma = {1, 2, 3, 0};
    weft::permute_tokens_by_dessin_simd(tokens_in.data(), sigma.data(), NUM_TOKENS, DIM, tokens_out.data());

    // Token 0 should have value from token 1 (2.0f)
    assert(tokens_out[0] == 2.0f);
    assert(tokens_out[1 * DIM] == 3.0f);
    assert(tokens_out[2 * DIM] == 4.0f);
    assert(tokens_out[3 * DIM] == 1.0f);

    std::cout << "  Permuted token values successfully verified." << std::endl;
    std::cout << "[PASS] All Grothendieck Dessins d'Enfants invariants verified." << std::endl;
    return 0;
}
