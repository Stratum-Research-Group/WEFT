#include "csrc/cli/weft_numa_ring_buffer.h"
#include <iostream>
#include <cassert>

struct SpeculativeTokenDraft {
    uint32_t token_id;
    float draft_prob;
    uint32_t layer_idx;
};

int main() {
    weft::NUMARingBuffer<SpeculativeTokenDraft, 128> ring;

    assert(ring.empty());
    assert(ring.size() == 0);

    for (uint32_t i = 0; i < 50; ++i) {
        bool ok = ring.push({i, 0.95f, 1});
        assert(ok);
    }

    assert(ring.size() == 50);

    SpeculativeTokenDraft item;
    for (uint32_t i = 0; i < 50; ++i) {
        bool ok = ring.pop(item);
        assert(ok);
        assert(item.token_id == i);
    }

    assert(ring.empty());

    std::cout << "[PASS] NUMA-Aware Zero-Copy Ring Buffer Unit Test passed.\n";
    return 0;
}
