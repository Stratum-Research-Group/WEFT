#include <iostream>
#include <vector>
#include <cassert>
#include "csrc/cli/weft_shared_kv_allocator.h"

int main() {
    std::cout << "[TEST] Running test_shared_kv_allocator..." << std::endl;

    const size_t pages = 128;
    weft::SharedKVAllocator alloc(pages, 4096);

    assert(alloc.available_pages() == pages);

    // Tenant 101 allocates 10 pages
    std::vector<int32_t> t1_pages;
    for (int i = 0; i < 10; ++i) {
        int32_t p = alloc.allocate_page(101);
        assert(p >= 0);
        t1_pages.push_back(p);
    }
    assert(alloc.available_pages() == pages - 10);

    // Tenant 102 tries to access Tenant 101's page buffer -> must fail
    uint8_t* illicit_buf = alloc.get_page_buffer(t1_pages[0], 102);
    assert(illicit_buf == nullptr);

    // Tenant 101 accesses own buffer -> succeeds
    uint8_t* legit_buf = alloc.get_page_buffer(t1_pages[0], 101);
    assert(legit_buf != nullptr);
    legit_buf[0] = 42;
    assert(legit_buf[0] == 42);

    // Tenant 102 tries to free Tenant 101's page -> must fail
    bool unauth_free = alloc.free_page(t1_pages[0], 102);
    assert(!unauth_free);

    // Tenant 101 frees all pages
    for (int32_t p : t1_pages) {
        bool ok = alloc.free_page(p, 101);
        assert(ok);
    }
    assert(alloc.available_pages() == pages);

    std::cout << "[PASS] test_shared_kv_allocator succeeded." << std::endl;
    return 0;
}
