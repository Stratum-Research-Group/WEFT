#include "../csrc/kernels/avx512_paged_attention.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

int main() {
    std::cout << "=== Test Multi-Threaded AVX-512 Paged Attention (OpenMP) ===" << std::endl;

    size_t head_dim = 64;
    size_t num_pages = 64; // 1024 tokens
    size_t total_tokens = num_pages * weft::AVX512PagedAttention::PAGE_SIZE;

    std::vector<float> q(head_dim, 0.5f);
    std::vector<float> pages_k(total_tokens * head_dim, 0.1f);
    std::vector<float> pages_v(total_tokens * head_dim, 1.0f);
    std::vector<float> out(head_dim, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 100; ++iter) {
        weft::AVX512PagedAttention::paged_attention_forward(
            q.data(), pages_k.data(), pages_v.data(), out.data(), num_pages, head_dim
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(end - start).count();

    std::cout << "Execution time for 100 iterations of 1024-token paged attention: " << sec << " s" << std::endl;
    // With uniform values of 1.0f, the attention output across dimension should be 1.0f
    assert(std::abs(out[0] - 1.0f) < 1e-4f);
    std::cout << "[PASS] Multi-Threaded Paged Attention Verified." << std::endl;

    return 0;
}
