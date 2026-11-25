#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <numeric>
#include <immintrin.h>
#include <omp.h>

// Measures memory streaming bandwidth and dot-product throughput across cache levels on Zen 4:
// L1 (~32 KB), L2 (~1 MB), L3 (~32 MB), and DDR5 (>64 MB)
int main() {
    std::cout << "[BENCHMARK] Starting AVX-512 Cache-Oblivious Tiling Sweep (AMD Ryzen 5 7600X)..." << std::endl;

    std::vector<size_t> sizes_bytes = {
        16 * 1024,        // 16 KB (L1)
        256 * 1024,       // 256 KB (L2)
        8 * 1024 * 1024,  // 8 MB (L3)
        64 * 1024 * 1024  // 64 MB (DDR5)
    };

    std::vector<std::string> cache_tiers = {"L1 Cache", "L2 Cache", "L3 Cache", "DDR5 Main Memory"};
    std::ofstream out_json("results/zen4_cache_tiling.json");
    out_json << "{\n  \"processor\": \"AMD Ryzen 5 7600X (6C/12T)\",\n  \"benchmarks\": [\n";

    for (size_t idx = 0; idx < sizes_bytes.size(); ++idx) {
        size_t bytes = sizes_bytes[idx];
        size_t num_floats = bytes / sizeof(float);
        std::vector<float> a(num_floats, 1.0f);
        std::vector<float> b(num_floats, 0.5f);
        std::vector<float> c(num_floats, 0.0f);

        // Warmup
        for (size_t i = 0; i < num_floats; ++i) c[i] = a[i] + b[i];

        int iters = (bytes <= 256 * 1024) ? 50000 : ((bytes <= 8 * 1024 * 1024) ? 2000 : 200);
        auto t0 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int it = 0; it < iters; ++it) {
            size_t i = 0;
            #if defined(__AVX512F__)
            for (; i + 16 <= num_floats; i += 16) {
                __m512 va = _mm512_loadu_ps(a.data() + i);
                __m512 vb = _mm512_loadu_ps(b.data() + i);
                __m512 vc = _mm512_fmadd_ps(va, vb, va);
                _mm512_storeu_ps(c.data() + i, vc);
            }
            #endif
            for (; i < num_floats; ++i) {
                c[i] = a[i] * b[i] + a[i];
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
        double total_bytes = static_cast<double>(bytes) * 3.0 * iters; // 2 reads + 1 write
        double gb_per_sec = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed_sec;
        double gflops = (static_cast<double>(num_floats) * 2.0 * iters / 1e9) / elapsed_sec;

        std::cout << "  - " << cache_tiers[idx] << " (" << bytes / 1024 << " KB): "
                  << gb_per_sec << " GB/s | " << gflops << " GFLOPS" << std::endl;

        out_json << "    {\n"
                 << "      \"tier\": \"" << cache_tiers[idx] << "\",\n"
                 << "      \"size_kb\": " << (bytes / 1024) << ",\n"
                 << "      \"bandwidth_gbps\": " << gb_per_sec << ",\n"
                 << "      \"gflops\": " << gflops << "\n"
                 << "    }" << (idx + 1 < sizes_bytes.size() ? "," : "") << "\n";
    }

    out_json << "  ]\n}\n";
    out_json.close();
    std::cout << "[DONE] Saved results to results/zen4_cache_tiling.json" << std::endl;
    return 0;
}
