#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_motivic_dt_bps.h"

int main() {
    std::cout << "[TEST] Running Motivic Donaldson-Thomas Invariants & BPS Bitboards Invariant Test..." << std::endl;

    const size_t NUM_BLOCKS = 64;
    std::vector<uint64_t> bitboards(NUM_BLOCKS, 0ULL);

    // 16 active blocks with non-zero BPS state count
    for (size_t b = 0; b < 16; ++b) {
        bitboards[b] = 0xAAAAAAAAAAAAAAAAULL; // 32 bits set
    }
    // 48 inert blocks with Omega = 0

    std::vector<uint64_t> compacted_out(NUM_BLOCKS, 0ULL);
    weft::BPSCompactionProfile prof = weft::compact_bps_bitboards(
        bitboards.data(), NUM_BLOCKS, compacted_out.data()
    );

    std::cout << "  Total blocks: " << prof.total_blocks << std::endl;
    std::cout << "  Active blocks: " << prof.active_blocks << " (expected: 16)" << std::endl;
    std::cout << "  Total BPS states: " << prof.total_bps_states << " (expected: 16 * 32 = 512)" << std::endl;
    std::cout << "  Index compression ratio: " << prof.index_compression_ratio << "x" << std::endl;

    assert(prof.total_blocks == 64);
    assert(prof.active_blocks == 16);
    assert(prof.total_bps_states == 512);
    assert(prof.index_compression_ratio == 4.0f);

    std::cout << "[PASS] All Motivic Donaldson-Thomas Invariants & BPS Bitboards invariants verified." << std::endl;
    return 0;
}
