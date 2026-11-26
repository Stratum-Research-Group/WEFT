// tools/weft_bench_lif_neuron.cpp
// AVX2-accelerated Leaky Integrate-and-Fire (LIF) and Exponential Decay Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

struct LIFConfig {
    float tau_m = 20.0f;       // membrane time constant (ms)
    float v_rest = -70.0f;     // resting potential (mV)
    float v_reset = -75.0f;    // reset potential (mV)
    float v_thresh = -50.0f;   // action potential threshold (mV)
    float dt = 0.5f;           // timestep (ms)
    float decay = std::exp(-0.5f / 20.0f); // exp(-dt / tau_m)
};

// Benchmark AVX2 Leaky Integrate-and-Fire vector simulation
void run_lif_avx2_bench(size_t num_neurons, size_t steps) {
    std::vector<float> v_mem(num_neurons, -70.0f);
    std::vector<float> currents(num_neurons, 25.0f); // injected current sufficient to reach threshold (-70 + 25 = -45 > -50)
    std::vector<uint8_t> spikes(num_neurons, 0);

    LIFConfig cfg;
    __m256 v_decay = _mm256_set1_ps(cfg.decay);
    __m256 v_rest = _mm256_set1_ps(cfg.v_rest);
    __m256 v_thresh = _mm256_set1_ps(cfg.v_thresh);
    __m256 v_reset = _mm256_set1_ps(cfg.v_reset);
    __m256 one_minus_decay = _mm256_set1_ps(1.0f - cfg.decay);

    size_t total_spikes = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_neurons; i += 8) {
            __m256 v = _mm256_loadu_ps(&v_mem[i]);
            __m256 I = _mm256_loadu_ps(&currents[i]);

            // V[t+1] = V[t] * decay + (V_rest + I) * (1 - decay)
            __m256 target_v = _mm256_add_ps(v_rest, I);
            __m256 updated_v = _mm256_fmadd_ps(target_v, one_minus_decay, _mm256_mul_ps(v, v_decay));

            // Spike check: V >= V_thresh
            __m256 spike_mask = _mm256_cmp_ps(updated_v, v_thresh, _CMP_GE_OQ);

            // Reset fired neurons to V_reset, keep others at updated_v
            __m256 next_v = _mm256_blendv_ps(updated_v, v_reset, spike_mask);
            _mm256_storeu_ps(&v_mem[i], next_v);

            int mask_bits = _mm256_movemask_ps(spike_mask);
            total_spikes += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_neurons) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 LIF Neuron Microbenchmark" << std::endl;
    std::cout << "  Neurons:        " << num_neurons << std::endl;
    std::cout << "  Timesteps:      " << steps << " (" << steps * cfg.dt << " ms sim)" << std::endl;
    std::cout << "  Elapsed Time:   " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:     " << std::fixed << std::setprecision(2) << millions_per_sec << " Million neuron-steps/sec" << std::endl;
    std::cout << "  Spikes Fired:   " << total_spikes << std::endl;
    std::cout << "  Mean Firing Rate:" << std::fixed << std::setprecision(2) 
              << (static_cast<double>(total_spikes) / (num_neurons * (steps * cfg.dt / 1000.0))) << " Hz" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t neurons = 131072; // 128k neurons
    size_t steps = 1000;      // 1000 steps (500 ms)
    if (argc > 1) neurons = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_lif_avx2_bench(neurons, steps);
    return 0;
}
