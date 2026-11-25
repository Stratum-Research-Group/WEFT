// Copyright (c) 2026 Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <cassert>
#include "../csrc/cli/weft_numa_token_streamer.h"

int main() {
    std::cout << "[TEST] Running NUMA Token Streamer Ring Buffer Test..." << std::endl;

    weft::NumaTokenStreamer streamer;
    assert(streamer.pending_count() == 0);

    for (uint32_t i = 1; i <= 50; ++i) {
        bool ok = streamer.push_token(i * 10);
        assert(ok);
    }
    assert(streamer.pending_count() == 50);

    for (uint32_t i = 1; i <= 50; ++i) {
        uint32_t tok = 0;
        bool ok = streamer.pop_token(tok);
        assert(ok);
        assert(tok == i * 10);
    }
    assert(streamer.pending_count() == 0);

    std::cout << "[PASS] NUMA Token Streamer Ring Buffer verified successfully." << std::endl;
    return 0;
}
