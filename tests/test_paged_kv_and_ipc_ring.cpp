#include <iostream>
#include <cassert>
#include <vector>
#include "../csrc/cli/weft_paged_kv.h"
#include "../csrc/cli/weft_ipc_ring.h"

int main() {
    std::cout << "[TEST] Running Weft Paged KV Cache & IPC Ring Verification..." << std::endl;

    // 1. Verify Paged KV Cache
    weft::WeftPagedKVCache kv(64, 4, 2);
    std::vector<weft::WeftKVBlock> blocks;
    std::vector<float> dummy_key(64, 1.25f);
    std::vector<float> dummy_val(64, -0.75f);

    for (size_t t = 0; t < 35; ++t) {
        kv.append_token(0, dummy_key.data(), dummy_val.data(), blocks);
    }
    // 35 tokens with 16 tokens/block should yield ceil(35/16) = 3 blocks
    assert(blocks.size() == 3);
    assert(blocks[0].num_tokens == 16);
    assert(blocks[1].num_tokens == 16);
    assert(blocks[2].num_tokens == 3);
    std::cout << "[PASS] Paged KV Cache allocation verified (3 blocks for 35 tokens)." << std::endl;

    // 2. Verify IPC Ring Buffer
    weft::WeftIPCRingBuffer ring;
    assert(ring.enqueue_draft(1042, 0.95f));
    assert(ring.enqueue_draft(1043, 0.88f));

    int32_t tok;
    float conf;
    assert(ring.dequeue_verify(tok, conf));
    assert(tok == 1042 && std::abs(conf - 0.95f) < 1e-5f);
    assert(ring.dequeue_verify(tok, conf));
    assert(tok == 1043 && std::abs(conf - 0.88f) < 1e-5f);
    assert(!ring.dequeue_verify(tok, conf)); // Empty now
    std::cout << "[PASS] Lock-Free IPC Ring Buffer enqueue/dequeue verified." << std::endl;

    std::cout << "[ALL PASSED] All Paged KV & IPC Ring tests successful." << std::endl;
    return 0;
}
