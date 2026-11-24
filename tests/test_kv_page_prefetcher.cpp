#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "csrc/cli/weft_kv_page_prefetcher.h"

int main() {
    std::cout << "[Test] Initializing KV Page Prefetcher SIMD Test..." << std::endl;

    const size_t page_elements = 64;
    alignas(64) std::vector<float> page1(page_elements, 1.0f);
    alignas(64) std::vector<float> page2(page_elements, 2.0f);
    alignas(64) std::vector<float> dst(page_elements, 0.0f);

    for (size_t i = 0; i < page_elements; ++i) {
        page1[i] = static_cast<float>(i + 1);
        page2[i] = static_cast<float>((i + 1) * 2);
    }

    const void* future_pages[2] = { page1.data(), page2.data() };

    // Prefetch future pages
    weft::avx512_prefetch_kv_pages(future_pages, 2, page_elements * sizeof(float));

    // Stream copy page1 to dst
    weft::avx512_stream_copy_page(page1.data(), dst.data(), page_elements);

    for (size_t i = 0; i < page_elements; ++i) {
        assert(std::fabs(dst[i] - page1[i]) < 1e-5f);
    }

    std::cout << "[Test] Page Prefetcher & Streaming Copy Succeeded!" << std::endl;
    return 0;
}
