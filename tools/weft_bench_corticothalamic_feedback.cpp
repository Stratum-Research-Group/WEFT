// tools/weft_bench_corticothalamic_feedback.cpp
// AVX2 SIMD Corticothalamic Layer VI Feedback Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_corticothalamic_avx2_bench(size_t num_channels, size_t steps) {
    std::vector<float> i_center(num_channels, 12.0f);
    std::vector<float> i_surround(num_channels, 8.0f);
    std::vector<float> out_center(num_channels, 0.0f);
    std::vector<float> out_surround(num_channels, 0.0f);

    float l6_drive = 0.85f;
    float gamma_l6 = 0.45f;
    float alpha_trn = 0.80f;

    float gain_c = 1.0f + gamma_l6 * l6_drive;
    float gain_s = std::max(0.05f, 1.0f - alpha_trn * l6_drive);

    __m256 v_gain_c = _mm256_set1_ps(gain_c);
    __m256 v_gain_s = _mm256_set1_ps(gain_s);

    size_t high_contrast_events = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_channels; i += 8) {
            __m256 v_c = _mm256_loadu_ps(&i_center[i]);
            __m256 v_s = _mm256_loadu_ps(&i_surround[i]);

            __m256 out_c = _mm256_mul_ps(v_c, v_gain_c);
            __m256 out_s = _mm256_mul_ps(v_s, v_gain_s);

            _mm256_storeu_ps(&out_center[i], out_c);
            _mm256_storeu_ps(&out_surround[i], out_s);

            // Contrast ratio: out_c / out_s > 3.0
            __m256 v_ratio = _mm256_div_ps(out_c, _mm256_add_ps(out_s, _mm256_set1_ps(1e-6f)));
            __m256 mask = _mm256_cmp_ps(v_ratio, _mm256_set1_ps(3.0f), _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(mask);
            high_contrast_events += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_channels) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Corticothalamic Layer VI Feedback Microbenchmark" << std::endl;
    std::cout << "  Receptive Field Channels: " << num_channels << std::endl;
    std::cout << "  Timesteps:                " << steps << std::endl;
    std::cout << "  Elapsed Time:             " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:               " << std::fixed << std::setprecision(2) << millions_per_sec << " Million channel-steps/sec" << std::endl;
    std::cout << "  High-Contrast Sharpened:  " << high_contrast_events << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t channels = 131072; // 128k RF channels
    size_t steps = 1000;
    if (argc > 1) channels = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_corticothalamic_avx2_bench(channels, steps);
    return 0;
}
