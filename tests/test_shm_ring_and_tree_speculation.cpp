#include <iostream>
#include <cassert>
#include "../csrc/cli/weft_shm_ring.h"

int main() {
    std::cout << "[TEST] Running Weft POSIX Shared Memory Ring Buffer Verification..." << std::endl;

    // Test owner creation and worker attachment
    weft::WeftShmRingBuffer ring_owner(true);
    assert(ring_owner.enqueue_draft(50256, 0.98f));
    assert(ring_owner.enqueue_draft(50257, 0.91f));

    int32_t tok;
    float conf;
    assert(ring_owner.dequeue_verify(tok, conf));
    assert(tok == 50256 && std::abs(conf - 0.98f) < 1e-4f);
    assert(ring_owner.dequeue_verify(tok, conf));
    assert(tok == 50257 && std::abs(conf - 0.91f) < 1e-4f);
    assert(!ring_owner.dequeue_verify(tok, conf));

    std::cout << "[PASS] POSIX Shm Ring Buffer verified successfully." << std::endl;
    return 0;
}
