// tools/weft_bench_basal_ganglia.cpp
// AVX2 SIMD Basal Ganglia Multi-Compartment Attractor Selection & Hyperdirect Brake Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

struct BasalGangliaChannelState {
    float cortex_cue;      // [0, 1] cortical attractor drive
    float d1_msn;          // Direct pathway MSN
    float d2_msn;          // Indirect pathway MSN
    float gpe;             // Globus Pallidus external
    float stn;             // Subthalamic Nucleus
    float gpi;             // Globus Pallidus internal / SNr
    float thalamus;        // Thalamocortical relay
};

void run_basal_ganglia_avx2_bench(size_t num_channels, size_t steps) {
    // 50% winning channels with high D1 (0.9), 50% suppressed with lower D1 (0.2)
    std::vector<float> d1(num_channels, 0.2f);
    for (size_t i = 0; i < num_channels / 2; ++i) d1[i] = 0.95f;
    std::vector<float> d2(num_channels, 0.2f);
    std::vector<float> stn(num_channels, 0.05f);
    std::vector<float> gpi(num_channels, 1.0f);
    std::vector<float> thalamus(num_channels, 0.0f);

    __m256 v_baseline_gpi = _mm256_set1_ps(1.0f);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_stn_weight = _mm256_set1_ps(1.5f);
    __m256 v_da = _mm256_set1_ps(0.95f); // High Phasic dopamine burst
    __m256 v_one_minus_da = _mm256_set1_ps(1.0f - 0.95f);

    size_t disinhibited_attractors = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_channels; i += 8) {
            __m256 v_d1 = _mm256_loadu_ps(&d1[i]);
            __m256 v_d2 = _mm256_loadu_ps(&d2[i]);
            __m256 v_stn = _mm256_loadu_ps(&stn[i]);

            // D1 disinhibition: d1 * da
            __m256 d1_term = _mm256_mul_ps(v_d1, v_da);
            // D2 indirect drive: d2 * (1 - da)
            __m256 d2_term = _mm256_mul_ps(v_d2, v_one_minus_da);
            // STN hyperdirect drive: stn * 1.5
            __m256 stn_term = _mm256_mul_ps(v_stn, v_stn_weight);

            // GPi = max(0, baseline - d1_term + d2_term + stn_term)
            __m256 net_gpi = _mm256_add_ps(_mm256_sub_ps(v_baseline_gpi, d1_term), _mm256_add_ps(d2_term, stn_term));
            net_gpi = _mm256_max_ps(v_zero, net_gpi);
            _mm256_storeu_ps(&gpi[i], net_gpi);

            // Thalamic aperture = max(0, min(1, 1 - GPi))
            __m256 thal = _mm256_max_ps(v_zero, _mm256_min_ps(v_one, _mm256_sub_ps(v_one, net_gpi)));
            _mm256_storeu_ps(&thalamus[i], thal);

            // Threshold aperture > 0.45 for attractor release
            __m256 gate_mask = _mm256_cmp_ps(thal, _mm256_set1_ps(0.45f), _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(gate_mask);
            disinhibited_attractors += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_channels) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Basal Ganglia Attractor Selection Microbenchmark" << std::endl;
    std::cout << "  Attractor Channels: " << num_channels << std::endl;
    std::cout << "  Timesteps:          " << steps << std::endl;
    std::cout << "  Elapsed Time:       " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:         " << std::fixed << std::setprecision(2) << millions_per_sec << " Million channel-steps/sec" << std::endl;
    std::cout << "  Attractors Released:" << disinhibited_attractors << std::endl;
    std::cout << "  Disinhibition Rate: " << std::fixed << std::setprecision(2) 
              << (100.0 * disinhibited_attractors / total_evaluations) << " %" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t channels = 65536; // 64k cortical attractor channels
    size_t steps = 1000;
    if (argc > 1) channels = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_basal_ganglia_avx2_bench(channels, steps);
    return 0;
}
