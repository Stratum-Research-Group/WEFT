#include "csrc/cli/weft_shm_broadcast_bus.h"
#include <iostream>
#include <cassert>

int main() {
    weft::SHMBroadcastBus<128> bus;

    uint64_t last_seq = 0;
    uint32_t token = 0;
    float prob = 0.0f;
    uint32_t model = 0;

    assert(!bus.read_latest(last_seq, token, prob, model));

    // Broadcast token 42 from model 1
    bus.broadcast(42, 0.99f, 1);

    bool ok = bus.read_latest(last_seq, token, prob, model);
    assert(ok);
    assert(token == 42);
    assert(prob == 0.99f);
    assert(model == 1);

    // Reading again should return false (no new tokens)
    assert(!bus.read_latest(last_seq, token, prob, model));

    std::cout << "[PASS] Shared Memory Broadcast Bus Unit Test passed.\n";
    return 0;
}
