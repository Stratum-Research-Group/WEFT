// Vectorized microbenchmark for Lateral Habenula (LHb) Anti-Reward Signaling & Dopamine Silencing
// Evaluates AVX2 + FMA SIMD throughput of negative RPE computation, LHb activation, RMTg GABA transmission, and VTA dopamine pause.
// Part of the WEFT Monorepo benchmark suite (Target #389).

#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

struct alignas(32) HabenulaBatch {
    static constexpr size_t N = 1000000;
    std::vector<float> expected_reward;
    std::vector<float> actual_outcome;
    std::vector<float> aversive_stress;
    std::vector<float> tonic_baseline;
    std::vector<float> output_nrpe;
    std::vector<float> output_lhb_rate;
    std::vector<float> output_rmtg_gaba;
    std::vector<float> output_vta_da;

    HabenulaBatch()
        : expected_reward(N), actual_outcome(N), aversive_stress(N), tonic_baseline(N),
          output_nrpe(N), output_lhb_rate(N), output_rmtg_gaba(N), output_vta_da(N) {
        for (size_t i = 0; i < N; ++i) {
            expected_reward[i] = 10.0f + (static_cast<float>(i % 80));
            actual_outcome[i] = (i % 3 == 0) ? 0.0f : expected_reward[i]; // 1/3 omission
            aversive_stress[i] = (i % 5 == 0) ? 40.0f : 5.0f; // Stress bursts
            tonic_baseline[i] = 8.0f;
        }
    }
};

void benchmark_habenula_avx2(HabenulaBatch& batch) {
    const size_t n = HabenulaBatch::N;
    const float* __restrict exp_rew = batch.expected_reward.data();
    const float* __restrict act_out = batch.actual_outcome.data();
    const float* __restrict stress = batch.aversive_stress.data();
    const float* __restrict base = batch.tonic_baseline.data();
    float* __restrict out_nrpe = batch.output_nrpe.data();
    float* __restrict out_lhb = batch.output_lhb_rate.data();
    float* __restrict out_rmtg = batch.output_rmtg_gaba.data();
    float* __restrict out_da = batch.output_vta_da.data();

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_c_nrpe = _mm256_set1_ps(0.45f);
    const __m256 v_c_stress = _mm256_set1_ps(0.35f);
    const __m256 v_c_rmtg_base = _mm256_set1_ps(5.0f);
    const __m256 v_c_rmtg_slope = _mm256_set1_ps(0.85f);
    const __m256 v_da_max = _mm256_set1_ps(5.0f);
    const __m256 v_c_da_slope = _mm256_set1_ps(0.075f);
    const __m256 v_rmtg_offset = _mm256_set1_ps(10.0f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_exp = _mm256_loadu_ps(exp_rew + i);
        __m256 v_act = _mm256_loadu_ps(act_out + i);
        __m256 v_stress = _mm256_loadu_ps(stress + i);
        __m256 v_base = _mm256_loadu_ps(base + i);

        // nRPE = max(0.0, exp - act)
        __m256 v_diff = _mm256_sub_ps(v_exp, v_act);
        __m256 v_nrpe = _mm256_max_ps(v_zero, v_diff);

        // Rate_LHb = base + 0.45 * nRPE + 0.35 * stress
        __m256 v_lhb = _mm256_fmadd_ps(v_c_nrpe, v_nrpe, v_base);
        v_lhb = _mm256_fmadd_ps(v_c_stress, v_stress, v_lhb);

        // Rate_RMTg = 5.0 + 0.85 * Rate_LHb
        __m256 v_rmtg = _mm256_fmadd_ps(v_c_rmtg_slope, v_lhb, v_c_rmtg_base);

        // DA_rate = max(0.0, 5.0 - 0.075 * (RMTg - 10.0))
        __m256 v_rmtg_delta = _mm256_sub_ps(v_rmtg, v_rmtg_offset);
        __m256 v_da_suppressed = _mm256_fnmadd_ps(v_c_da_slope, v_rmtg_delta, v_da_max);
        __m256 v_da = _mm256_max_ps(v_zero, v_da_suppressed);

        _mm256_storeu_ps(out_nrpe + i, v_nrpe);
        _mm256_storeu_ps(out_lhb + i, v_lhb);
        _mm256_storeu_ps(out_rmtg + i, v_rmtg);
        _mm256_storeu_ps(out_da + i, v_da);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "WEFT AVX2+FMA Lateral Habenula Anti-Reward Benchmark\n";
    std::cout << "========================================================\n";

    HabenulaBatch batch;

    // Warmup
    for (int w = 0; w < 5; ++w) {
        benchmark_habenula_avx2(batch);
    }

    constexpr int ITERS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        benchmark_habenula_avx2(batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double throughput = (static_cast<double>(HabenulaBatch::N) / (avg_ms / 1000.0)) / 1e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Batch Size:           " << HabenulaBatch::N << " elements\n";
    std::cout << "Total Runtime (" << ITERS << "x):   " << total_ms << " ms\n";
    std::cout << "Latency per batch:    " << avg_ms << " ms\n";
    std::cout << "Throughput:           " << throughput << " Melem/s\n";

    // Verification sample
    std::cout << "\nSample Verification at index 0 (Exp=" << batch.expected_reward[0] 
              << ", Act=" << batch.actual_outcome[0] << ", Stress=" << batch.aversive_stress[0] << "):\n";
    std::cout << "  Negative RPE:    " << batch.output_nrpe[0] << "\n";
    std::cout << "  LHb Firing Rate: " << batch.output_lhb_rate[0] << " Hz\n";
    std::cout << "  RMTg GABA Out:   " << batch.output_rmtg_gaba[0] << " Hz\n";
    std::cout << "  VTA DA Firing:   " << batch.output_vta_da[0] << " Hz\n";
    std::cout << "========================================================\n";

    return 0;
}
