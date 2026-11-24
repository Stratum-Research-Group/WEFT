#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include "../csrc/cli/weft_lockfree_page_table.h"

int main() {
    std::cout << "[TEST] Running Weft Lock-Free Page Pool Verification..." << std::endl;

    const size_t pool_size = 64;
    weft::WeftLockFreePagePool pool(pool_size, 64);

    // 1. Basic Sequential Alloc & Free
    uint32_t p1, p2, p3;
    assert(pool.allocate_page(p1));
    assert(pool.allocate_page(p2));
    assert(pool.allocate_page(p3));
    assert(p1 != p2 && p2 != p3);
    std::cout << "[PASS] Basic sequential page allocations verified (" << p1 << ", " << p2 << ", " << p3 << ")." << std::endl;

    pool.free_page(p1);
    pool.free_page(p2);
    pool.free_page(p3);

    // 2. Concurrent Multi-Threaded Allocation Test
    std::vector<std::thread> workers;
    std::atomic<size_t> success_count{0};

    for (int t = 0; t < 8; ++t) {
        workers.emplace_back([&pool, &success_count]() {
            for (int i = 0; i < 5; ++i) {
                uint32_t pid;
                if (pool.allocate_page(pid)) {
                    success_count++;
                    pool.free_page(pid);
                }
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    assert(success_count == 40);
    std::cout << "[PASS] Multi-threaded concurrent allocation verified (40 CAS alloc/free ops)." << std::endl;

    std::cout << "[ALL PASSED] Lock-Free Page Pool tests successful." << std::endl;
    return 0;
}
