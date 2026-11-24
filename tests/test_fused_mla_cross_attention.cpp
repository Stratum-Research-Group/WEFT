#include <iostream>
#include <cassert>
#include <vector>
#include "../csrc/cli/weft_fused_mla.h"

int main() {
    std::cout << "[TEST] Running Weft Fused MLA Cross-Attention Verification..." << std::endl;

    weft::WeftMLAPagedKVCache cache(1);
    std::vector<weft::WeftMLAKVBlock> blocks;

    std::vector<float> latent_vec(weft::MLA_LATENT_KV_DIM, 0.5f);
    latent_vec[0] = -1.0f;
    latent_vec[511] = 2.0f;
    cache.append_mla_token(latent_vec.data(), blocks);
    assert(!blocks.empty());

    std::vector<float> transformed_query(512, 1.0f);
    std::vector<float> query_rope(64, 0.5f);
    std::vector<float> rope_key(64, 0.5f);

    float score = weft::fused_mla_total_score(transformed_query.data(), query_rope.data(),
                                              blocks[0], 0, rope_key.data());

    std::cout << "[INFO] Computed Fused MLA attention logit: " << score << std::endl;
    assert(!std::isnan(score));
    assert(!std::isinf(score));
    assert(score > 0.0f);
    std::cout << "[PASS] Fused MLA Cross-Attention verified successfully." << std::endl;

    return 0;
}
