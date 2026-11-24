#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../csrc/cli/weft_perfectoid_shimura.h"

int main() {
    std::cout << "[TEST] Running Perfectoid Shimura Varieties & Hodge-Tate Map Invariant Test..." << std::endl;

    const size_t TOTAL_PARAMS = 100;
    const size_t NUM_WORKERS = 4;
    std::vector<float> weights(TOTAL_PARAMS, 1.0f);

    size_t accumulated_len = 0;
    std::vector<weft::PerfectoidSliceResult> slices;

    for (size_t w = 0; w < NUM_WORKERS; ++w) {
        weft::PerfectoidSliceResult slice = weft::partition_hodge_tate_slice(
            weights.data(), TOTAL_PARAMS, w, NUM_WORKERS
        );
        std::cout << "  Worker " << w << ": start=" << slice.slice_start
                  << ", len=" << slice.slice_len
                  << ", norm=" << slice.local_frobenius_norm << std::endl;

        // Disjointness verification
        assert(slice.slice_start == accumulated_len);
        accumulated_len += slice.slice_len;
        assert(slice.local_frobenius_norm > 0.0f);
        slices.push_back(slice);
    }

    // Partition completeness: total length must equal TOTAL_PARAMS
    assert(accumulated_len == TOTAL_PARAMS);

    std::cout << "[PASS] All Perfectoid Shimura Varieties & Hodge-Tate Map invariants verified." << std::endl;
    return 0;
}
