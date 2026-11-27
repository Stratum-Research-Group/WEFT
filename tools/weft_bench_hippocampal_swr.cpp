// tools/weft_bench_hippocampal_swr.cpp
// Native AVX2/FMA SIMD benchmark for Hippocampal CA1 Sharp Wave-Ripple (SWR) Kinetics
// Computes PV+ basket cell GABAergic conductance, ripple oscillation frequency, phase coherence (PLV), and replay fidelity

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

static constexpr size_t N_CHANNELS = 65536;
static constexpr int N_ITERATIONS = 1000;

void benchmark_hippocampal_swr_avx2(
    const float* __restrict__ pv_drive,
    const float* __restrict__ ca3_input,
    float* __restrict__ out_freq,
    float* __restrict__ out_plv,
    float* __restrict__ out_fidelity,
    float* __restrict__ out_spikes,
    size_t n
) {
    const __m256 v_base_freq = _mm256_set1_ps(150.0f);
    const __m256 v_freq_slope = _mm256_set1_ps(0.60f); // 60 / 100
    const __m256 v_pv_weight = _mm256_set1_ps(0.0070f); // 0.70 / 100
    const __m256 v_ca3_weight = _mm256_set1_ps(0.0030f); // 0.30 / 100
    const __m256 v_min_plv = _mm256_set1_ps(0.20f);
    const __m256 v_max_plv = _mm256_set1_ps(0.98f);
    const __m256 v_fidelity_gain = _mm256_set1_ps(98.0f);
    const __m256 v_base_spikes = _mm256_set1_ps(2.0f);
    const __m256 v_spike_slope = _mm256_set1_ps(0.055f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_pv = _mm256_loadu_ps(pv_drive + i);
        __m256 v_ca3 = _mm256_loadu_ps(ca3_input + i);

        // freq = 150.0 + pv * 0.60
        __m256 v_f = _mm256_fmadd_ps(v_pv, v_freq_slope, v_base_freq);
        _mm256_storeu_ps(out_freq + i, v_f);

        // raw_plv = pv * 0.0070 + ca3 * 0.0030
        __m256 v_pv_part = _mm256_mul_ps(v_pv, v_pv_weight);
        __m256 v_raw_plv = _mm256_fmadd_ps(v_ca3, v_ca3_weight, v_pv_part);
        __m256 v_plv = _mm256_max_ps(v_min_plv, _mm256_min_ps(v_max_plv, v_raw_plv));
        _mm256_storeu_ps(out_plv + i, v_plv);

        // fidelity = plv * 98.0
        __m256 v_fid = _mm256_mul_ps(v_plv, v_fidelity_gain);
        _mm256_storeu_ps(out_fidelity + i, v_fid);

        // spikes = 2.0 + ca3 * 0.055
        __m256 v_spk = _mm256_fmadd_ps(v_ca3, v_spike_slope, v_base_spikes);
        _mm256_storeu_ps(out_spikes + i, v_spk);
    }
}

int main() {
    std::vector<float> pv(N_CHANNELS, 85.0f);
    std::vector<float> ca3(N_CHANNELS, 80.0f);
    std::vector<float> freq(N_CHANNELS);
    std::vector<float> plv(N_CHANNELS);
    std::vector<float> fidelity(N_CHANNELS);
    std::vector<float> spikes(N_CHANNELS);

    // Warmup
    for (int i = 0; i < 50; ++i) {
        benchmark_hippocampal_swr_avx2(
            pv.data(), ca3.data(),
            freq.data(), plv.data(), fidelity.data(), spikes.data(),
            N_CHANNELS
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERATIONS; ++i) {
        benchmark_hippocampal_swr_avx2(
            pv.data(), ca3.data(),
            freq.data(), plv.data(), fidelity.data(), spikes.data(),
            N_CHANNELS
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = total_us / N_ITERATIONS;
    double throughput_mch_s = (static_cast<double>(N_CHANNELS) / (avg_us * 1e-6)) / 1e6;

    std::cout << "[AVX2-FMA Benchmark] Hippocampal CA1 Sharp Wave-Ripple (SWR) Kinetics" << std::endl;
    std::cout << "Channels: " << N_CHANNELS << " | Iterations: " << N_ITERATIONS << std::endl;
    std::cout << "Avg latency per iteration: " << avg_us << " us" << std::endl;
    std::cout << "Throughput: " << throughput_mch_s << " Million channels/sec" << std::endl;
    std::cout << "Sample verification [Channel 0]:" << std::endl;
    std::cout << "  PV+ Drive: " << pv[0] << "% | CA3 Input: " << ca3[0] << "%" << std::endl;
    std::cout << "  Ripple Freq: " << freq[0] << " Hz | Coherence PLV: " << plv[0] << std::endl;
    std::cout << "  Replay Fidelity: " << fidelity[0] << "% | Pyramidal Spikes: " << spikes[0] << " / SWR" << std::endl;

    return 0;
}
