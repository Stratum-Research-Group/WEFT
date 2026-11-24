#include <iostream>
#include <cassert>
#include <vector>
#include "../csrc/cli/weft_mla_paged_kv.h"

int main() {
    std::cout << "[TEST] Running MLA Paged KV Cache & Early-Exit Verification..." << std::endl;

    weft::WeftMLAPagedKVCache mla_cache(1);
    std::vector<weft::WeftMLAKVBlock> blocks;

    std::vector<float> latent_vec(weft::MLA_LATENT_KV_DIM, 0.75f);
    latent_vec[0] = -1.2f;
    latent_vec[575] = 2.4f;

    // Append 20 tokens (should span 2 blocks: 16 + 4)
    for (size_t i = 0; i < 20; ++i) {
        mla_cache.append_mla_token(latent_vec.data(), blocks);
    }

    assert(blocks.size() == 2);
    assert(blocks[0].num_tokens == 16);
    assert(blocks[1].num_tokens == 4);
    assert(blocks[0].packed_latent_kv.size() == 16 * 144);
    std::cout << "[PASS] MLA Paged KV Cache allocation verified (144 bytes/token)." << std::endl;

    // Test early-exit pruning
    assert(blocks[0].active_mask == 0xFFFF);
    mla_cache.prune_branch(blocks[0], 0x0004); // Prune branch containing node 2
    assert((blocks[0].active_mask & 0x0004) == 0);
    assert((blocks[0].active_mask & 0x0001) != 0);
    std::cout << "[PASS] Early-exit branch pruning mask verified." << std::endl;

    std::cout << "[ALL PASSED] MLA Paged KV & Early-Exit tests successful." << std::endl;
    return 0;
}
