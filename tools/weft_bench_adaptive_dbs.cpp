#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

/**
 * WEFT AVX2 / FMA Microbenchmark: Target #399
 * Adaptive Closed-Loop DBS (aDBS): Phase-Specific Desynchronization & Beta-Burst Truncation
 * Collaborative Research Target with Kevin W. McCairn, Ph.D.
 * 
 * Computes:
 * - Real-time burst duration tracking & adaptive stimulation trigger policy
 * - Phase-specific pulse delivery (150° - 210° window)
 * - Post-intervention burst truncation & LFP beta power quenching
 * - Cumulative energy consumption vs continuous cDBS
 */

void compute_adbs_closed_loop_avx2(
    const float* __restrict__ raw_burst_duration,
    const float* __restrict__ lfp_phase_deg,
    float* __restrict__ truncated_duration,
    float* __restrict__ stim_intensity,
    float* __restrict__ beta_power,
    float* __restrict__ adbs_energy,
    size_t n
) {
    const __m256 v_thresh = _mm256_set1_ps(250.0f);
    const __m256 v_phase_low = _mm256_set1_ps(150.0f);
    const __m256 v_phase_high = _mm256_set1_ps(210.0f);
    const __m256 v_trunc_opt = _mm256_set1_ps(275.0f);
    const __m256 v_trunc_subopt = _mm256_set1_ps(325.0f);
    const __m256 v_stim_opt = _mm256_set1_ps(2.5f);
    const __m256 v_stim_subopt = _mm256_set1_ps(0.8f);
    const __m256 v_zeros = _mm256_setzero_ps();
    const __m256 v_base_power = _mm256_set1_ps(20.0f);
    const __m256 v_power_scale = _mm256_set1_ps(0.12f);
    const __m256 v_energy_step = _mm256_set1_ps(0.35f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 v_dur = _mm256_loadu_ps(&raw_burst_duration[i]);
        __m256 v_phase = _mm256_loadu_ps(&lfp_phase_deg[i]);

        // Check if burst exceeds trigger threshold (250 ms)
        __m256 trigger_mask = _mm256_cmp_ps(v_dur, v_thresh, _CMP_GT_OQ);

        // Check if phase is in optimal desynchronizing window [150, 210]
        __m256 phase_ge = _mm256_cmp_ps(v_phase, v_phase_low, _CMP_GE_OQ);
        __m256 phase_le = _mm256_cmp_ps(v_phase, v_phase_high, _CMP_LE_OQ);
        __m256 opt_phase_mask = _mm256_and_ps(phase_ge, phase_le);

        // Select truncated duration: opt (275 ms) vs subopt (325 ms)
        __m256 trunc_target = _mm256_blendv_ps(v_trunc_subopt, v_trunc_opt, opt_phase_mask);
        __m256 final_dur = _mm256_blendv_ps(v_dur, trunc_target, trigger_mask);
        _mm256_storeu_ps(&truncated_duration[i], final_dur);

        // Select stimulation intensity: opt (2.5 mA) vs subopt (0.8 mA) vs 0.0 mA
        __m256 stim_level = _mm256_blendv_ps(v_stim_subopt, v_stim_opt, opt_phase_mask);
        __m256 final_stim = _mm256_blendv_ps(v_zeros, stim_level, trigger_mask);
        _mm256_storeu_ps(&stim_intensity[i], final_stim);

        // Beta power = 20.0 + 0.12 * truncated_duration
        __m256 power = _mm256_fmadd_ps(final_dur, v_power_scale, v_base_power);
        _mm256_storeu_ps(&beta_power[i], power);

        // Energy consumption relative unit
        __m256 energy = _mm256_blendv_ps(v_zeros, v_energy_step, trigger_mask);
        _mm256_storeu_ps(&adbs_energy[i], energy);
    }
}

int main() {
    const size_t N = 1000000; // 1M state samples
    std::vector<float> raw_burst_duration(N), lfp_phase_deg(N);
    std::vector<float> truncated_duration(N), stim_intensity(N), beta_power(N), adbs_energy(N);

    // Populate synthetic trace representing parkinsonian burst distribution
    for (size_t i = 0; i < N; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(N);
        // Phase rotates 0-360
        lfp_phase_deg[i] = std::fmod(frac * 50000.0f, 360.0f);
        // Bursts oscillate between short (80-180 ms) and long pathological (300-600 ms)
        float sin_val = std::sin(frac * 200.0f);
        raw_burst_duration[i] = (sin_val > 0.0f) ? (150.0f + 450.0f * sin_val) : (80.0f + 100.0f * (1.0f + sin_val));
    }

    // Warmup
    compute_adbs_closed_loop_avx2(
        raw_burst_duration.data(), lfp_phase_deg.data(),
        truncated_duration.data(), stim_intensity.data(),
        beta_power.data(), adbs_energy.data(), N
    );

    // Benchmark loop
    const int WARMUP_RUNS = 20;
    const int BENCH_RUNS = 100;
    for (int r = 0; r < WARMUP_RUNS; ++r) {
        compute_adbs_closed_loop_avx2(
            raw_burst_duration.data(), lfp_phase_deg.data(),
            truncated_duration.data(), stim_intensity.data(),
            beta_power.data(), adbs_energy.data(), N
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < BENCH_RUNS; ++r) {
        compute_adbs_closed_loop_avx2(
            raw_burst_duration.data(), lfp_phase_deg.data(),
            truncated_duration.data(), stim_intensity.data(),
            beta_power.data(), adbs_energy.data(), N
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / BENCH_RUNS;
    double throughput_melem = (static_cast<double>(N) / (avg_ms / 1000.0)) / 1e6;

    // Cumulative duty cycle
    double total_adbs_energy = 0.0;
    for (size_t i = 0; i < N; ++i) total_adbs_energy += adbs_energy[i];
    double avg_duty = (total_adbs_energy / static_cast<double>(N)) / 0.35;
    double battery_savings = (1.0 - avg_duty) * 100.0;

    std::cout << "=================================================================\n";
    std::cout << "WEFT AVX2/FMA Benchmark: Target #399 Adaptive Closed-Loop DBS\n";
    std::cout << "=================================================================\n";
    std::cout << "Vector Size:           " << N << " elements\n";
    std::cout << "Average Latency:       " << avg_ms << " ms per million state evaluations\n";
    std::cout << "Throughput:            " << throughput_melem << " Melem/s\n";
    std::cout << "Stimulation Duty:      " << avg_duty * 100.0 << " % (vs 100% in cDBS)\n";
    std::cout << "Energy Savings:        " << battery_savings << " % reduction in battery consumption\n";
    std::cout << "Sample Trace:\n";
    std::cout << "  [Normal Burst]  Raw: " << raw_burst_duration[10] << " ms -> Truncated: " << truncated_duration[10] << " ms | Stim: " << stim_intensity[10] << " mA\n";
    std::cout << "  [Pathol Burst]  Raw: " << raw_burst_duration[1250] << " ms -> Truncated: " << truncated_duration[1250] << " ms | Stim: " << stim_intensity[1250] << " mA\n";
    std::cout << "=================================================================\n";

    return 0;
}
