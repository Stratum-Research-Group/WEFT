#include "csrc/cli/weft_tropical_moduli_stable_curves.h"
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    constexpr size_t num_heads = 8;
    constexpr size_t head_dim = 64;
    std::vector<float> head_queries(num_heads * head_dim, 0.0f);

    // Initialize distinct heads
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t d = 0; d < head_dim; ++d) {
            head_queries[h * head_dim + d] = static_cast<float>(h + 1) * 0.5f + static_cast<float>(d) * 0.01f;
        }
    }

    // Make head 3 an exact duplicate of head 1 (metric distance 0 -> redundant)
    for (size_t d = 0; d < head_dim; ++d) {
        head_queries[3 * head_dim + d] = head_queries[1 * head_dim + d];
    }

    uint32_t active_mask = weft::tropical_moduli_route_heads_avx512(head_queries.data(), num_heads, head_dim, 0.05f);

    // Head 0 and Head 1 must be active
    assert(active_mask & (1u << 0));
    assert(active_mask & (1u << 1));
    // Head 3 must be pruned
    assert(!(active_mask & (1u << 3)));

    std::cout << "[PASS] Tropical Moduli Stable Curves Routing Unit Test passed (Active mask: 0x" 
              << std::hex << active_mask << std::dec << ").\n";
    return 0;
}
