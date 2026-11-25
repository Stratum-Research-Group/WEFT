#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <omp.h>
#include "../csrc/kernels/avx512_paged_attention.h"

int main() {
    std::cout << "=== OpenMP + AVX-512 Thread-Scaling Benchmark on Zen 4 ===" << std::endl;

    size_t head_dim = 64;
    size_t num_pages = 128; // 2048 tokens
    size_t total_tokens = num_pages * weft::AVX512PagedAttention::PAGE_SIZE;

    std::vector<float> q(head_dim, 0.5f);
    std::vector<float> pages_k(total_tokens * head_dim, 0.1f);
    std::vector<float> pages_v(total_tokens * head_dim, 1.0f);
    std::vector<float> out(head_dim, 0.0f);

    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12};
    std::ofstream out_file("results/zen4_thread_scaling.json");
    out_file << "{\n  \"cpu\": \"AMD Ryzen 5 7600X (6C/12T)\",\n  \"tokens\": " << total_tokens << ",\n  \"scaling\": [\n";

    double base_time = 1.0;

    for (size_t i = 0; i < thread_counts.size(); ++i) {
        int t = thread_counts[i];
        omp_set_num_threads(t);

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; ++iter) {
            weft::AVX512PagedAttention::paged_attention_forward(
                q.data(), pages_k.data(), pages_v.data(), out.data(), num_pages, head_dim
            );
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        if (t == 1) base_time = elapsed;
        double speedup = base_time / elapsed;

        std::cout << "Threads: " << t << ", Elapsed: " << elapsed << " s, Speedup: " << speedup << "x" << std::endl;

        out_file << "    {\"threads\": " << t << ", \"elapsed_seconds\": " << elapsed << ", \"speedup\": " << speedup << "}"
                 << (i + 1 < thread_counts.size() ? ",\n" : "\n");
    }

    out_file << "  ]\n}\n";
    out_file.close();
    std::cout << "[PASS] Thread-scaling results written to results/zen4_thread_scaling.json" << std::endl;
    return 0;
}
