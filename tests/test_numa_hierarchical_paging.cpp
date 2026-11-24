// test_numa_hierarchical_paging.cpp - Verification for Paper #126
#include "csrc/cli/weft_numa_paged_kv.h"
#include <iostream>
#include <cassert>

using namespace weft;

int main() {
    std::cout << "[TEST] Initializing test_numa_hierarchical_paging..." << std::endl;

    constexpr size_t TOTAL_PAGES = 128;
    WeftNUMAPagedKVPool pool(TOTAL_PAGES, /*local_node=*/0);

    size_t hot_count = 0, cold_count = 0;
    pool.get_tier_counts(hot_count, cold_count);
    assert(hot_count == TOTAL_PAGES);
    assert(cold_count == 0);
    std::cout << "[TEST] Initial hot pages: " << hot_count << ", cold pages: " << cold_count << std::endl;

    // Access pages 0 through 31 frequently
    for (size_t iter = 0; iter < 50; ++iter) {
        for (size_t p = 0; p < 32; ++p) {
            pool.get_page(p);
        }
    }

    // Evict cold pages (pages with access counter < 20) to remote NUMA node 1
    size_t demoted = pool.evict_cold_pages(/*cold_threshold=*/20, /*remote_node=*/1);
    std::cout << "[TEST] Pages demoted to cold tier: " << demoted << std::endl;
    assert(demoted == TOTAL_PAGES - 32);

    pool.get_tier_counts(hot_count, cold_count);
    assert(hot_count == 32);
    assert(cold_count == demoted);
    assert(hot_count + cold_count == TOTAL_PAGES);

    // Now access a cold page (page 50) and promote it back to local node
    std::cout << "[TEST] Accessing and promoting cold page 50..." << std::endl;
    bool promoted = pool.migrate_page_to_local(50);
    assert(promoted);

    pool.get_tier_counts(hot_count, cold_count);
    assert(hot_count == 33);
    assert(cold_count == TOTAL_PAGES - 33);

    std::cout << "[TEST] Page count conservation across tiers verified: "
              << hot_count << " (local) + " << cold_count << " (remote) = " << TOTAL_PAGES << std::endl;
    std::cout << "[TEST] All NUMA hierarchical paging checks passed successfully!" << std::endl;
    return 0;
}
