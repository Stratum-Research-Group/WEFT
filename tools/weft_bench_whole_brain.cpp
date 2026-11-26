// tools/weft_bench_whole_brain.cpp
// AVX2 SIMD Whole-Brain Macro-Circuit Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_whole_brain_avx2_bench(size_t num_units, size_t steps) {
    std::vector<float> v_ctx(num_units, 1.0f);
    std::vector<float> v_thal(num_units, 2.0f);
    std::vector<float> s_cl(num_units, 1.5f);
    std::vector<float> v_bg(num_units, 2.5f);
    std::vector<float> v_cer(num_units, 1.8f);
    std::vector<float> v_out(num_units, 0.0f);

    __m256 w_thal = _mm256_set1_ps(0.35f);
    __m256 w_cl = _mm256_set1_ps(0.25f);
    __m256 w_bg = _mm256_set1_ps(0.30f);
    __m256 w_cer = _mm256_set1_ps(0.20f);
    __m256 w_ctx = _mm256_set1_ps(0.50f);
    __m256 v_max = _mm256_set1_ps(10.0f);
    __m256 v_min = _mm256_set1_ps(-10.0f);

    size_t active_nodes = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_units; i += 8) {
            __m256 thal = _mm256_loadu_ps(&v_thal[i]);
            __m256 cl = _mm256_loadu_ps(&s_cl[i]);
            __m256 bg = _mm256_loadu_ps(&v_bg[i]);
            __m256 cer = _mm256_loadu_ps(&v_cer[i]);
            __m256 ctx = _mm256_loadu_ps(&v_ctx[i]);

            // Sum weighted subcortical drives
            __m256 net = _mm256_fmadd_ps(thal, w_thal, _mm256_mul_ps(cl, w_cl));
            net = _mm256_fmadd_ps(bg, w_bg, net);
            net = _mm256_fmadd_ps(cer, w_cer, net);
            net = _mm256_fmadd_ps(ctx, w_ctx, net);

            // Bounded saturation
            net = _mm256_min_ps(v_max, _mm256_max_ps(v_min, net));
            _mm256_storeu_ps(&v_out[i], net);

            __m256 mask = _mm256_cmp_ps(net, _mm256_set1_ps(2.0f), _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(mask);
            active_nodes += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_units) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Whole-Brain Macro-Circuit Microbenchmark" << std::endl;
    std::cout << "  Multi-Compartment Units: " << num_units << std::endl;
    std::cout << "  Timesteps:               " << steps << std::endl;
    std::cout << "  Elapsed Time:            " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:              " << std::fixed << std::setprecision(2) << millions_per_sec << " Million unit-steps/sec" << std::endl;
    std::cout << "  Synchronized States:     " << active_nodes << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t units = 262144; // 256k macro-circuit units
    size_t steps = 1000;
    if (argc > 1) units = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_whole_brain_avx2_bench(units, steps);
    return 0;
}
