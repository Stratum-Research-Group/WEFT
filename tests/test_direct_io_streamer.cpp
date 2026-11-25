#include "csrc/cli/weft_direct_io_streamer.h"
#include <iostream>
#include <cassert>

int main() {
    void* buf = weft::DirectIOStreamer::allocate_aligned_buffer(8192);
    assert(buf != nullptr);
    assert(reinterpret_cast<uintptr_t>(buf) % 4096 == 0);

    weft::DirectIOStreamer::stream_non_temporal_prefetch(buf, 8192);

    weft::DirectIOStreamer::free_aligned_buffer(buf);

    std::cout << "[PASS] Direct I/O Weight Streamer Unit Test passed.\n";
    return 0;
}
