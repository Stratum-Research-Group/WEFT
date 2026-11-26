// tools/weft_bench_thalamic_relay.cpp
// AVX2 SIMD Dual-Compartment Thalamic Relay Microbenchmark (Burst vs Tonic Dynamics)
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_thalamic_relay_avx2_bench(size_t num_neurons, size_t steps) {
    // Half neurons hyperpolarized (-75 mV, burst primed), half depolarized (-55 mV, tonic)
    std::vector<float> v_mem(num_neurons, -55.0f);
    for (size_t i = 0; i < num_neurons / 2; ++i) v_mem[i] = -75.0f;
    std::vector<float> h_gate(num_neurons, 0.1f);
    for (size_t i = 0; i < num_neurons / 2; ++i) h_gate[i] = 0.85f; // De-inactivated T-type channels
    std::vector<float> i_stim(num_neurons, 12.0f); // Injected excitatory pulse

    __m256 v_thresh = _mm256_set1_ps(-50.0f);
    __m256 v_reset = _mm256_set1_ps(-70.0f);
    __m256 v_decay = _mm256_set1_ps(0.95f);
    __m256 v_burst_boost = _mm256_set1_ps(25.0f); // Rebound LTS calcium depolarizing current
    __m256 v_zero = _mm256_setzero_ps();

    size_t burst_spikes = 0;
    size_t tonic_spikes = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_neurons; i += 8) {
            __m256 v = _mm256_loadu_ps(&v_mem[i]);
            __m256 h = _mm256_loadu_ps(&h_gate[i]);
            __m256 stim = _mm256_loadu_ps(&i_stim[i]);

            // If h > 0.5, apply calcium burst boost
            __m256 is_burst_primed = _mm256_cmp_ps(h, _mm256_set1_ps(0.5f), _CMP_GT_OQ);
            __m256 extra_drive = _mm256_blendv_ps(v_zero, v_burst_boost, is_burst_primed);

            // Integrate V
            __m256 updated_v = _mm256_fmadd_ps(v, v_decay, _mm256_add_ps(stim, extra_drive));

            // Spike detection
            __m256 spike_mask = _mm256_cmp_ps(updated_v, v_thresh, _CMP_GE_OQ);
            __m256 next_v = _mm256_blendv_ps(updated_v, v_reset, spike_mask);
            _mm256_storeu_ps(&v_mem[i], next_v);

            int mask_bits = _mm256_movemask_ps(spike_mask);
            if (i < num_neurons / 2) {
                burst_spikes += __builtin_popcount(mask_bits);
            } else {
                tonic_spikes += __builtin_popcount(mask_bits);
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_neurons) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Thalamocortical Relay Microbenchmark" << std::endl;
    std::cout << "  Neurons:        " << num_neurons << std::endl;
    std::cout << "  Timesteps:      " << steps << std::endl;
    std::cout << "  Elapsed Time:   " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:     " << std::fixed << std::setprecision(2) << millions_per_sec << " Million neuron-steps/sec" << std::endl;
    std::cout << "  LTS Burst Spikes: " << burst_spikes << std::endl;
    std::cout << "  Tonic Spikes:     " << tonic_spikes << std::endl;
    std::cout << "  Burst Amplification Ratio: " << std::fixed << std::setprecision(2)
              << (tonic_spikes > 0 ? static_cast<double>(burst_spikes) / tonic_spikes : 0.0) << "x" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t neurons = 131072; // 128k thalamocortical relay units
    size_t steps = 1000;
    if (argc > 1) neurons = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_thalamic_relay_avx2_bench(neurons, steps);
    return 0;
}
