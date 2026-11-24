#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_donaldson_thomas_kv.h"

int main() {
    std::cout << "[TEST] Running Donaldson-Thomas KV Compaction Test..." << std::endl;

    // Test 1: Zero weight for curvature below threshold
    float w0 = weft::compute_behrend_weight(0.4f, 1.0f);
    float w1 = weft::compute_behrend_weight(2.5f, 1.0f);
    std::cout << "  Behrend weights: sub-threshold=" << w0 << ", super-threshold=" << w1 << std::endl;
    assert(w0 == 0.0f);
    assert(w1 == 2.0f);

    // Test 2: In-register KV cache compaction
    const size_t NUM_TOKENS = 8;
    const size_t DIM = 16;
    std::vector<float> kv_in(NUM_TOKENS * DIM, 1.0f);
    std::vector<float> kv_out(NUM_TOKENS * DIM, 0.0f);
    std::vector<float> curvature = {0.2f, 1.5f, 0.8f, 2.1f, 0.1f, 3.0f, 0.5f, 1.2f};

    size_t retained = weft::compact_kv_cache_donaldson_thomas_simd(
        kv_in.data(), curvature.data(), NUM_TOKENS, DIM, 1.0f, kv_out.data()
    );

    std::cout << "  Retained tokens (tau=1.0): " << retained << " / " << NUM_TOKENS << std::endl;
    // Indices with curvature >= 1.0 are 1 (1.5), 3 (2.1), 5 (3.0), 7 (1.2) -> 4 tokens
    assert(retained == 4);

    for (size_t i = 0; i < retained * DIM; ++i) {
        assert(kv_out[i] == 1.0f);
    }

    std::cout << "[PASS] All Donaldson-Thomas KV Compaction invariants verified." << std::endl;
    return 0;
}
