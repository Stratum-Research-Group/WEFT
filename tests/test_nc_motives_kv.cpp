#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_nc_motives_kv.h"

int main() {
    std::cout << "[TEST] Running Non-Commutative Motives & CY KV-Cache Invariant Test..." << std::endl;

    const size_t NUM_TOKENS = 64;
    const size_t HEAD_DIM = 16;
    std::vector<float> keys(NUM_TOKENS * HEAD_DIM, 0.0f);
    std::vector<float> values(NUM_TOKENS * HEAD_DIM, 0.0f);

    // Active tokens: tokens 0..15 have high energy
    for (size_t t = 0; t < 16; ++t) {
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            keys[t * HEAD_DIM + d] = 1.0f;
            values[t * HEAD_DIM + d] = 1.0f;
        }
    }
    // Inert tokens: tokens 16..63 have near-zero energy
    for (size_t t = 16; t < NUM_TOKENS; ++t) {
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            keys[t * HEAD_DIM + d] = 0.001f;
            values[t * HEAD_DIM + d] = 0.001f;
        }
    }

    float threshold = 0.5f;
    weft::NCKVProfile prof = weft::prune_kv_cache_nc_motives(keys.data(), values.data(), NUM_TOKENS, HEAD_DIM, threshold);

    std::cout << "  Active tokens: " << prof.active_tokens << " (expected 16)" << std::endl;
    std::cout << "  Pruned tokens: " << prof.pruned_tokens << " (expected 48)" << std::endl;
    std::cout << "  Compression ratio: " << prof.compression_ratio << "x" << std::endl;
    std::cout << "  Cyclic trace norm: " << prof.cyclic_trace_norm << std::endl;

    assert(prof.active_tokens == 16);
    assert(prof.pruned_tokens == 48);
    assert(prof.compression_ratio == 4.0f);
    assert(prof.cyclic_trace_norm > 0.0f);

    std::cout << "[PASS] All Non-Commutative Motives & CY KV-Cache invariants verified." << std::endl;
    return 0;
}
