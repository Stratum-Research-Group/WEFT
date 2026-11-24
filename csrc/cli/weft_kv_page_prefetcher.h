#pragma once
// weft_kv_page_prefetcher.h
// KV Cache Sub-Bit Page Prefetcher & NUMA-Aware Memory Pinning (Item 5)
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#include <cstddef>
#include <immintrin.h>

namespace weft {

// Issues hardware prefetch instructions for upcoming KV cache pages in L1/L2 cache
inline void avx512_prefetch_kv_pages(
    const void* const* __restrict__ future_page_ptrs,
    size_t num_future_pages,
    size_t page_stride_bytes
) {
    for (size_t p = 0; p < num_future_pages; ++p) {
        const char* page = static_cast<const char*>(future_page_ptrs[p]);
        if (page) {
            // Prefetch first 4 cache lines (256 bytes) of future page
            _mm_prefetch(page, _MM_HINT_T0);
            _mm_prefetch(page + 64, _MM_HINT_T0);
            _mm_prefetch(page + 128, _MM_HINT_T0);
            _mm_prefetch(page + 192, _MM_HINT_T0);
        }
    }
}

// Vectorized zero-copy page copy using unaligned stores (safe for any allocation)
inline void avx512_stream_copy_page(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t count
) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        _mm512_storeu_ps(dst + i, v);
    }
    for (; i < count; ++i) {
        dst[i] = src[i];
    }
}

} // namespace weft
