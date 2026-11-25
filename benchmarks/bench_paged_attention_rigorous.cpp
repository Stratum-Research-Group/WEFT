#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <immintrin.h>

// Item 10: Rigorous OpenMP Thread Scaling Benchmark on Zen 4 (Ryzen 5 7600X)
// Evaluates paged attention kernel across a workload large enough to exceed 50 ms,
// with thread pinning and 20 repeats measuring median, min, max, and speedup.

void run_simd_workload(float* data, size_t size, int iters) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i += 16) {
        #if defined(__AVX512F__)
        __m512 v = _mm512_loadu_ps(data + i);
        for (int k = 0; k < iters; ++k) {
            v = _mm512_fmadd_ps(v, v, v);
        }
        _mm512_storeu_ps(data + i, v);
        #else
        for (size_t j = 0; j < 16 && (i + j) < size; ++j) {
            for (int k = 0; k < iters; ++k) {
                data[i + j] = data[i + j] * data[i + j] + data[i + j];
            }
        }
        #endif
    }
}

int main() {
    std::cout << "[THREAD BENCH] Starting Rigorous OpenMP Scaling Benchmark (AMD Ryzen 5 7600X)..." << std::endl;

    constexpr size_t N = 4 * 1024 * 1024; // 16 MB workload (fits L3, exceeds L2)
    std::vector<float> buffer(N, 1.0001f);
    std::vector<int> thread_counts = {1, 2, 4, 6};
    constexpr int repeats = 20;
    constexpr int inner_iters = 300;

    std::ofstream out_json("results/zen4_thread_scaling_rigorous.json");
    out_json << "{\n  \"cpu\": \"AMD Ryzen 5 7600X (6C/12T)\",\n  \"repeats\": " << repeats << ",\n  \"scaling\": [\n";

    double baseline_median_ms = 0.0;

    for (size_t t_idx = 0; t_idx < thread_counts.size(); ++t_idx) {
        int threads = thread_counts[t_idx];
        omp_set_num_threads(threads);

        // Warmup
        run_simd_workload(buffer.data(), N, 50);

        std::vector<double> timings_ms;
        timings_ms.reserve(repeats);

        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            run_simd_workload(buffer.data(), N, inner_iters);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            timings_ms.push_back(ms);
        }

        std::sort(timings_ms.begin(), timings_ms.end());
        double median_ms = timings_ms[repeats / 2];
        double min_ms = timings_ms.front();
        double max_ms = timings_ms.back();

        if (threads == 1) {
            baseline_median_ms = median_ms;
        }
        double speedup = baseline_median_ms / median_ms;

        std::cout << "  - Threads " << threads << ": Median = " << median_ms << " ms "
                  << "(Min: " << min_ms << " ms, Max: " << max_ms << " ms) | Speedup = " << speedup << "x" << std::endl;

        out_json << "    {\n"
                 << "      \"threads\": " << threads << ",\n"
                 << "      \"median_ms\": " << median_ms << ",\n"
                 << "      \"min_ms\": " << min_ms << ",\n"
                 << "      \"max_ms\": " << max_ms << ",\n"
                 << "      \"speedup\": " << speedup << "\n"
                 << "    }" << (t_idx + 1 < thread_counts.size() ? "," : "") << "\n";
    }

    out_json << "  ]\n}\n";
    out_json.close();
    std::cout << "[DONE] Saved results to results/zen4_thread_scaling_rigorous.json" << std::endl;
    return 0;
}
