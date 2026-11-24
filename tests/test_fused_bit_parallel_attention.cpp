#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "../csrc/cli/weft_fused_attention.h"

int main() {
    std::cout << "[TEST] Running Fused Bit-Parallel Attention Verification..." << std::endl;

    const size_t dim = 64;
    weft::WeftPagedKVCache kv(dim, 1, 1);
    std::vector<weft::WeftKVBlock> blocks;

    std::vector<float> key(dim);
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
        key[i] = static_cast<float>(i % 4) * 0.5f;
        query[i] = 1.0f;
    }

    kv.append_token(0, key.data(), key.data(), blocks);
    assert(!blocks.empty());

    float dot = weft::fused_bit_parallel_dot(query.data(), blocks[0], 0, dim);
    std::cout << "[INFO] Computed fused bit-parallel dot product: " << dot << std::endl;
    assert(!std::isnan(dot));
    assert(!std::isinf(dot));
    assert(dot > 0.0f);

    std::cout << "[PASS] Fused bit-parallel attention test successful." << std::endl;
    return 0;
}
