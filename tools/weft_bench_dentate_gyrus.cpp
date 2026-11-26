// tools/weft_bench_dentate_gyrus.cpp
// AVX2 SIMD Dentate Gyrus Sparse Pattern Separation Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_dentate_gyrus_avx2_bench(size_t num_granule_cells, size_t steps) {
    // 131,072 granule cells receiving random input drives
    std::vector<float> raw_drives(num_granule_cells);
    std::vector<float> sparse_out(num_granule_cells, 0.0f);

    for (size_t i = 0; i < num_granule_cells; ++i) {
        raw_drives[i] = static_cast<float>(rand() % 1000) / 100.0f; // [0, 10]
    }

    // High threshold to enforce extreme sparsity (< 2% active)
    float v_thresh = 9.80f;
    __m256 v_th = _mm256_set1_ps(v_thresh);
    __m256 v_zero = _mm256_setzero_ps();

    size_t active_granule_cells = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_granule_cells; i += 8) {
            __m256 v = _mm256_loadu_ps(&raw_drives[i]);

            // Sparse thresholding: max(0, v - v_thresh)
            __m256 diff = _mm256_sub_ps(v, v_th);
            __m256 out = _mm256_max_ps(v_zero, diff);
            _mm256_storeu_ps(&sparse_out[i], out);

            __m256 active_mask = _mm256_cmp_ps(out, v_zero, _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(active_mask);
            active_granule_cells += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_granule_cells) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;
    double sparsity_pct = (static_cast<double>(active_granule_cells) / total_evaluations) * 100.0;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Dentate Gyrus Sparse Separation Microbenchmark" << std::endl;
    std::cout << "  Granule Cells:  " << num_granule_cells << std::endl;
    std::cout << "  Timesteps:      " << steps << std::endl;
    std::cout << "  Elapsed Time:   " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:     " << std::fixed << std::setprecision(2) << millions_per_sec << " Million cell-steps/sec" << std::endl;
    std::cout << "  Active Spikes:  " << active_granule_cells << std::endl;
    std::cout << "  Sparsity Rate:  " << std::fixed << std::setprecision(2) << sparsity_pct << " % (Target: < 2.5%)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t cells = 131072; // 128k granule cells
    size_t steps = 1000;
    if (argc > 1) cells = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    srand(42);
    run_dentate_gyrus_avx2_bench(cells, steps);
    return 0;
}
