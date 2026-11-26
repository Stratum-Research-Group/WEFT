// Vectorized microbenchmark for Claustrum Cortical Synchronization & Salience Switching
// Evaluates AVX2 + FMA SIMD throughput of claustral burst scaling, feedforward interneuron inhibition, and network synchrony resets.
// Part of the WEFT Monorepo benchmark suite (Target #393).

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

struct alignas(32) ClaustrumBatch {
    static constexpr size_t N = 1000000;
    std::vector<float> salience_input;
    std::vector<float> task_drive;
    std::vector<float> claustral_baseline;
    std::vector<float> interneuron_gain;
    std::vector<float> output_claustral_rate;
    std::vector<float> output_feedforward_inh;
    std::vector<float> output_synchrony_index;
    std::vector<float> output_switch_prob;

    ClaustrumBatch()
        : salience_input(N), task_drive(N), claustral_baseline(N), interneuron_gain(N),
          output_claustral_rate(N), output_feedforward_inh(N), output_synchrony_index(N), output_switch_prob(N) {
        for (size_t i = 0; i < N; ++i) {
            salience_input[i] = 1.0f + (static_cast<float>(i % 50) / 10.0f); // 1 - 6
            task_drive[i] = 2.0f + (static_cast<float>(i % 40) / 10.0f);     // 2 - 6
            claustral_baseline[i] = 2.0f;
            interneuron_gain[i] = 0.5f;
        }
    }
};

void benchmark_claustrum_avx2(ClaustrumBatch& batch) {
    const size_t n = ClaustrumBatch::N;
    const float* __restrict sal = batch.salience_input.data();
    const float* __restrict task = batch.task_drive.data();
    const float* __restrict base = batch.claustral_baseline.data();
    const float* __restrict gain = batch.interneuron_gain.data();
    float* __restrict out_rate = batch.output_claustral_rate.data();
    float* __restrict out_inh = batch.output_feedforward_inh.data();
    float* __restrict out_sync = batch.output_synchrony_index.data();
    float* __restrict out_prob = batch.output_switch_prob.data();

    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_c_sal = _mm256_set1_ps(0.85f);
    const __m256 v_c_inh_scale = _mm256_set1_ps(1.8f);
    const __m256 v_sync_base = _mm256_set1_ps(0.10f);
    const __m256 v_sync_scale = _mm256_set1_ps(0.065f);
    const __m256 v_c_drive_scale = _mm256_set1_ps(0.80f);
    const __m256 v_c_inh_eff = _mm256_set1_ps(0.25f);
    const __m256 v_inh_offset = _mm256_set1_ps(3.0f);
    const __m256 v_half = _mm256_set1_ps(0.50f);
    const __m256 v_sig_slope = _mm256_set1_ps(0.20f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_sal = _mm256_loadu_ps(sal + i);
        __m256 v_task = _mm256_loadu_ps(task + i);
        __m256 v_base = _mm256_loadu_ps(base + i);
        __m256 v_gain = _mm256_loadu_ps(gain + i);

        // Rate_cl = base + 0.85 * sal
        __m256 v_rate = _mm256_fmadd_ps(v_c_sal, v_sal, v_base);

        // Feedforward Inh = 1.0 + gain * 1.8 * Rate_cl
        __m256 v_gain_eff = _mm256_mul_ps(v_gain, v_c_inh_scale);
        __m256 v_inh = _mm256_fmadd_ps(v_gain_eff, v_rate, v_one);

        // Synchrony = clamp(0.10 + 0.065 * Rate_cl, 0.0, 1.0)
        __m256 v_sync_raw = _mm256_fmadd_ps(v_sync_scale, v_rate, v_sync_base);
        __m256 v_sync = _mm256_max_ps(v_zero, _mm256_min_ps(v_one, v_sync_raw));

        // Switch Prob Proxy: clamp(0.5 + 0.20 * ((sal - task)*0.8 + 0.25*(inh - 3.0)), 0.0, 1.0)
        __m256 v_diff = _mm256_sub_ps(v_sal, v_task);
        __m256 v_net_drive = _mm256_mul_ps(v_diff, v_c_drive_scale);
        __m256 v_inh_contrib = _mm256_mul_ps(_mm256_sub_ps(v_inh, v_inh_offset), v_c_inh_eff);
        __m256 v_comb = _mm256_add_ps(v_net_drive, v_inh_contrib);
        __m256 v_prob_raw = _mm256_fmadd_ps(v_sig_slope, v_comb, v_half);
        __m256 v_prob = _mm256_max_ps(v_zero, _mm256_min_ps(v_one, v_prob_raw));

        _mm256_storeu_ps(out_rate + i, v_rate);
        _mm256_storeu_ps(out_inh + i, v_inh);
        _mm256_storeu_ps(out_sync + i, v_sync);
        _mm256_storeu_ps(out_prob + i, v_prob);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "WEFT AVX2+FMA Claustrum Synchronization Benchmark (T393)\n";
    std::cout << "========================================================\n";

    ClaustrumBatch batch;

    // Warmup
    for (int w = 0; w < 5; ++w) {
        benchmark_claustrum_avx2(batch);
    }

    constexpr int ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        benchmark_claustrum_avx2(batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double throughput = (static_cast<double>(ClaustrumBatch::N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch Size:           " << ClaustrumBatch::N << " elements\n";
    std::cout << "Total Runtime (" << ITERS << "x):   " << total_ms << " ms\n";
    std::cout << "Latency per batch:    " << avg_ms << " ms\n";
    std::cout << "Throughput:           " << throughput << " Melem/s\n";

    // Verification sample
    std::cout << "\nSample Verification at index 0 (Salience=" << batch.salience_input[0] 
              << ", Task=" << batch.task_drive[0] << "):\n";
    std::cout << "  Claustral Firing: " << batch.output_claustral_rate[0] << " Hz\n";
    std::cout << "  Feedforward Inh:  " << batch.output_feedforward_inh[0] << "x\n";
    std::cout << "  Synchrony Index:  " << (batch.output_synchrony_index[0] * 100.0f) << "%\n";
    std::cout << "  Task Switch Prob: " << (batch.output_switch_prob[0] * 100.0f) << "%\n";
    std::cout << "========================================================\n";

    return 0;
}
