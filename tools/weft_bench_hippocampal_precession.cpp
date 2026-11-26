// tools/weft_bench_hippocampal_precession.cpp
// AVX2 SIMD Hippocampal Place Cell Phase Precession Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_hippocampal_precession_avx2_bench(size_t num_cells, size_t steps) {
    std::vector<float> positions(num_cells, 0.5f);
    std::vector<float> field_centers(num_cells);
    std::vector<float> field_widths(num_cells, 0.8f);
    std::vector<float> probs(num_cells, 0.0f);

    for (size_t i = 0; i < num_cells; ++i) {
        field_centers[i] = static_cast<float>(i % 100) / 100.0f;
    }

    float theta_phase = 3.14159f; // Mid-theta cycle
    __m256 v_theta = _mm256_set1_ps(theta_phase);
    __m256 v_two_pi = _mm256_set1_ps(6.2831853f);
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_one = _mm256_set1_ps(1.0f);

    size_t active_spikes = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        float sim_pos = static_cast<float>(s % 1000) / 1000.0f;
        __m256 v_pos = _mm256_set1_ps(sim_pos);

        for (size_t i = 0; i < num_cells; i += 8) {
            __m256 v_center = _mm256_loadu_ps(&field_centers[i]);
            __m256 v_width = _mm256_loadu_ps(&field_widths[i]);

            // dx = x - (x_c - 0.5 * width)
            __m256 left_edge = _mm256_sub_ps(v_center, _mm256_mul_ps(v_half, v_width));
            __m256 dx = _mm256_sub_ps(v_pos, left_edge);
            __m256 u = _mm256_div_ps(dx, v_width);

            // In field: 0 <= u <= 1
            __m256 in_field = _mm256_and_ps(_mm256_cmp_ps(u, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(u, v_one, _CMP_LE_OQ));

            // Target phase: 0.95*2pi - u * 0.8*2pi
            __m256 target_phi = _mm256_sub_ps(_mm256_mul_ps(v_two_pi, _mm256_set1_ps(0.95f)),
                                              _mm256_mul_ps(u, _mm256_mul_ps(v_two_pi, _mm256_set1_ps(0.8f))));

            __m256 diff = _mm256_sub_ps(v_theta, target_phi);
            __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);

            // Spike threshold: abs_diff < 0.5 radians (~30 deg)
            __m256 spike_mask = _mm256_and_ps(in_field, _mm256_cmp_ps(abs_diff, _mm256_set1_ps(0.5f), _CMP_LT_OQ));
            int mask_bits = _mm256_movemask_ps(spike_mask);
            active_spikes += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_cells) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Hippocampal Phase Precession Microbenchmark" << std::endl;
    std::cout << "  Place Cells:    " << num_cells << std::endl;
    std::cout << "  Timesteps:      " << steps << std::endl;
    std::cout << "  Elapsed Time:   " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:     " << std::fixed << std::setprecision(2) << millions_per_sec << " Million cell-steps/sec" << std::endl;
    std::cout << "  Active Spikes:  " << active_spikes << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t cells = 65536; // 64k place cells
    size_t steps = 1000;
    if (argc > 1) cells = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_hippocampal_precession_avx2_bench(cells, steps);
    return 0;
}
