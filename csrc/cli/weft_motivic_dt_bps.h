#ifndef WEFT_MOTIVIC_DT_BPS_H
#define WEFT_MOTIVIC_DT_BPS_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct BPSCompactionProfile {
    size_t total_blocks;
    size_t active_blocks;
    size_t total_bps_states;
    float index_compression_ratio;
};

// Compacts sparse 64-bit bitboards using BPS state integer counts
// Evicts blocks with vanishing BPS count (Omega = 0) with zero metadata allocation
inline BPSCompactionProfile compact_bps_bitboards(
    const uint64_t* bitboard_masks,
    size_t num_blocks,
    uint64_t* compacted_masks_out
) {
    BPSCompactionProfile prof;
    prof.total_blocks = num_blocks;
    prof.active_blocks = 0;
    prof.total_bps_states = 0;

    size_t out_idx = 0;

    for (size_t b = 0; b < num_blocks; ++b) {
        uint64_t mask = bitboard_masks[b];
        uint32_t popcnt = static_cast<uint32_t>(__builtin_popcountll(mask));

        if (popcnt > 0) {
            compacted_masks_out[out_idx++] = mask;
            prof.active_blocks++;
            prof.total_bps_states += popcnt;
        }
    }

    prof.index_compression_ratio = (prof.active_blocks > 0) ?
        static_cast<float>(num_blocks) / static_cast<float>(prof.active_blocks) : 1.0f;

    return prof;
}

} // namespace weft

#endif // WEFT_MOTIVIC_DT_BPS_H
