// tools/weft_bench_neuromodulation.cpp
// AVX2 SIMD Cortical Multi-Transmitter Neuromodulatory Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_neuromodulation_avx2_bench(size_t num_neurons, size_t steps) {
    std::vector<float> i_ff(num_neurons, 5.0f);   // Feedforward drive
    std::vector<float> i_rec(num_neurons, 3.0f);  // Recurrent noise
    std::vector<float> i_eff(num_neurons, 0.0f);

    float ne_level = 0.65f; // Locus Coeruleus NE tone
    float ach_level = 0.70f; // Basal forebrain ACh tone

    float gain_ne = 1.0f + 1.8f * ne_level;
    float gain_rec = std::max(0.1f, 1.0f - 0.6f * ach_level);

    __m256 v_gain_ne = _mm256_set1_ps(gain_ne);
    __m256 v_gain_rec = _mm256_set1_ps(gain_rec);
    __m256 v_max = _mm256_set1_ps(50.0f);
    __m256 v_min = _mm256_set1_ps(-50.0f);

    size_t high_snr_channels = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_neurons; i += 8) {
            __m256 v_f = _mm256_loadu_ps(&i_ff[i]);
            __m256 v_r = _mm256_loadu_ps(&i_rec[i]);

            // I_eff = gain_ne * I_ff + gain_rec * I_rec
            __m256 v_net = _mm256_fmadd_ps(v_f, v_gain_ne, _mm256_mul_ps(v_r, v_gain_rec));

            // Clamp [-50, 50]
            v_net = _mm256_min_ps(v_max, _mm256_max_ps(v_min, v_net));
            _mm256_storeu_ps(&i_eff[i], v_net);

            __m256 active_mask = _mm256_cmp_ps(v_net, _mm256_set1_ps(10.0f), _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(active_mask);
            high_snr_channels += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_neurons) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Neuromodulatory Gain Microbenchmark" << std::endl;
    std::cout << "  Neurons:        " << num_neurons << std::endl;
    std::cout << "  Timesteps:      " << steps << std::endl;
    std::cout << "  Elapsed Time:   " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:     " << std::fixed << std::setprecision(2) << millions_per_sec << " Million neuron-steps/sec" << std::endl;
    std::cout << "  High-SNR Channels:" << high_snr_channels << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t neurons = 131072; // 128k neurons
    size_t steps = 1000;
    if (argc > 1) neurons = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_neuromodulation_avx2_bench(neurons, steps);
    return 0;
}
