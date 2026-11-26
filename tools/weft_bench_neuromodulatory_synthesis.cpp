// Vectorized microbenchmark for Central Neuromodulatory Convergence & Multi-Transmitter Synthesis
// Evaluates AVX2 + FMA SIMD throughput of concurrent 5-axis neuromodulation (ACh x NE x 5-HT x DA x LHb).
// Part of the WEFT Monorepo benchmark suite (Target #391 - Milestone Paper #200).

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

struct alignas(32) SynthesisBatch {
    static constexpr size_t N = 1000000;
    std::vector<float> ach_tone;
    std::vector<float> ne_rate;
    std::vector<float> serotonin_tone;
    std::vector<float> da_rpe;
    std::vector<float> lhb_stress;
    std::vector<float> output_cortical_gain;
    std::vector<float> output_net_da;
    std::vector<float> output_waiting_cap;
    std::vector<float> output_motor_readiness;

    SynthesisBatch()
        : ach_tone(N), ne_rate(N), serotonin_tone(N), da_rpe(N), lhb_stress(N),
          output_cortical_gain(N), output_net_da(N), output_waiting_cap(N), output_motor_readiness(N) {
        for (size_t i = 0; i < N; ++i) {
            ach_tone[i] = 1.0f + (static_cast<float>(i % 50) / 10.0f);
            ne_rate[i] = 1.0f + (static_cast<float>(i % 60) / 10.0f);
            serotonin_tone[i] = 0.5f + (static_cast<float>(i % 70) / 10.0f);
            da_rpe[i] = 0.5f + (static_cast<float>(i % 40) / 10.0f);
            lhb_stress[i] = (i % 4 == 0) ? 2.5f : 0.2f; // Stress bursts
        }
    }
};

void benchmark_synthesis_avx2(SynthesisBatch& batch) {
    const size_t n = SynthesisBatch::N;
    const float* __restrict ach = batch.ach_tone.data();
    const float* __restrict ne = batch.ne_rate.data();
    const float* __restrict sht = batch.serotonin_tone.data();
    const float* __restrict da = batch.da_rpe.data();
    const float* __restrict lhb = batch.lhb_stress.data();
    float* __restrict out_gain = batch.output_cortical_gain.data();
    float* __restrict out_da = batch.output_net_da.data();
    float* __restrict out_cap = batch.output_waiting_cap.data();
    float* __restrict out_ready = batch.output_motor_readiness.data();

    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_c_ach = _mm256_set1_ps(0.35f);
    const __m256 v_c_ne = _mm256_set1_ps(0.35f);
    const __m256 v_gain_max = _mm256_set1_ps(10.0f);
    const __m256 v_c_sht = _mm256_set1_ps(0.45f);
    const __m256 v_scale_readiness = _mm256_set1_ps(0.10f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_ach = _mm256_loadu_ps(ach + i);
        __m256 v_ne = _mm256_loadu_ps(ne + i);
        __m256 v_sht = _mm256_loadu_ps(sht + i);
        __m256 v_da = _mm256_loadu_ps(da + i);
        __m256 v_lhb = _mm256_loadu_ps(lhb + i);

        // Effective Cortical Gain: G = min(G_max, 1.0 + 0.35 * ACh + 0.35 * NE)
        __m256 v_gain_raw = _mm256_fmadd_ps(v_c_ach, v_ach, v_one);
        v_gain_raw = _mm256_fmadd_ps(v_c_ne, v_ne, v_gain_raw);
        __m256 v_gain = _mm256_min_ps(v_gain_max, v_gain_raw);

        // Net Consolidated Dopamine: DA_net = max(0.0, DA - LHb)
        __m256 v_da_diff = _mm256_sub_ps(v_da, v_lhb);
        __m256 v_net_da = _mm256_max_ps(v_zero, v_da_diff);

        // Waiting Capacity: P = 1.0 + 0.45 * 5HT
        __m256 v_cap = _mm256_fmadd_ps(v_c_sht, v_sht, v_one);

        // Motor Readiness: R = (G * DA_net) * 0.10
        __m256 v_readiness = _mm256_mul_ps(v_scale_readiness, _mm256_mul_ps(v_gain, v_net_da));

        _mm256_storeu_ps(out_gain + i, v_gain);
        _mm256_storeu_ps(out_da + i, v_net_da);
        _mm256_storeu_ps(out_cap + i, v_cap);
        _mm256_storeu_ps(out_ready + i, v_readiness);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "WEFT AVX2+FMA Neuromodulatory Synthesis Benchmark (P200)\n";
    std::cout << "========================================================\n";

    SynthesisBatch batch;

    // Warmup
    for (int w = 0; w < 5; ++w) {
        benchmark_synthesis_avx2(batch);
    }

    constexpr int ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        benchmark_synthesis_avx2(batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double throughput = (static_cast<double>(SynthesisBatch::N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch Size:           " << SynthesisBatch::N << " elements (5 transmitters/element)\n";
    std::cout << "Total Runtime (" << ITERS << "x):   " << total_ms << " ms\n";
    std::cout << "Latency per batch:    " << avg_ms << " ms\n";
    std::cout << "Throughput:           " << throughput << " Melem/s\n";

    // Verification sample
    std::cout << "\nSample Verification at index 0 (ACh=" << batch.ach_tone[0] 
              << ", NE=" << batch.ne_rate[0] << ", 5HT=" << batch.serotonin_tone[0]
              << ", DA=" << batch.da_rpe[0] << ", LHb=" << batch.lhb_stress[0] << "):\n";
    std::cout << "  Cortical Gain:   " << batch.output_cortical_gain[0] << "x\n";
    std::cout << "  Net Dopamine:    " << batch.output_net_da[0] << "\n";
    std::cout << "  Waiting Cap:     " << batch.output_waiting_cap[0] << "x\n";
    std::cout << "  Motor Readiness: " << batch.output_motor_readiness[0] << "\n";
    std::cout << "========================================================\n";

    return 0;
}
