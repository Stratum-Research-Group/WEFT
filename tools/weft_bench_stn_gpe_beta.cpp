#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for GPe Prototypic vs Arkypallidal Closed-Loop Resonator
// Evaluates:
// 1. Arkypallidal stop firing: arky = stop ? 55.0 : (8.0 + (100 - da) * 0.08)
// 2. Striatal motor inhibition: striatal_out = max(0, drive - (drive * arky) / 60)
// 3. STN-GPe Parkinsonian beta resonance frequency: f_res = 13.0 + (100 - da) * 0.17
// Target #383 in WEFT Verification Suite.

void benchmark_scalar_gpe(const std::vector<float>& da_tone,
                          const std::vector<float>& striatal_drive,
                          const std::vector<float>& stop_active,
                          std::vector<float>& arky_rate,
                          std::vector<float>& striatal_out,
                          std::vector<float>& beta_freq,
                          size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float da = da_tone[i];
        float drive = striatal_drive[i];
        bool stop = stop_active[i] > 0.5f;

        float arky = stop ? 55.0f : (8.0f + (100.0f - da) * 0.08f);
        arky_rate[i] = arky;

        float inhib = (drive * arky) / 60.0f;
        striatal_out[i] = std::fmax(0.0f, drive - inhib);

        float da_def = std::fmax(0.0f, 100.0f - da);
        beta_freq[i] = 13.0f + (da_def * 17.0f) / 100.0f;
    }
}

void benchmark_avx2_gpe(const float* da_tone,
                        const float* striatal_drive,
                        const float* stop_active,
                        float* arky_rate,
                        float* striatal_out,
                        float* beta_freq,
                        size_t n) {
    const __m256 v_100 = _mm256_set1_ps(100.0f);
    const __m256 v_55 = _mm256_set1_ps(55.0f);
    const __m256 v_8 = _mm256_set1_ps(8.0f);
    const __m256 v_0_08 = _mm256_set1_ps(0.08f);
    const __m256 v_13 = _mm256_set1_ps(13.0f);
    const __m256 v_0_17 = _mm256_set1_ps(0.17f);
    const __m256 v_inv_60 = _mm256_set1_ps(1.0f / 60.0f);
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_zero = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 da = _mm256_loadu_ps(da_tone + i);
        __m256 drive = _mm256_loadu_ps(striatal_drive + i);
        __m256 stop = _mm256_loadu_ps(stop_active + i);

        // da_deficit = max(0, 100 - da)
        __m256 da_def = _mm256_max_ps(v_zero, _mm256_sub_ps(v_100, da));

        // Baseline arky: 8.0 + da_def * 0.08
        __m256 base_arky = _mm256_fmadd_ps(da_def, v_0_08, v_8);
        __m256 stop_mask = _mm256_cmp_ps(stop, v_half, _CMP_GT_OQ);
        __m256 arky = _mm256_blendv_ps(base_arky, v_55, stop_mask);
        _mm256_storeu_ps(arky_rate + i, arky);

        // Striatal out: max(0, drive - drive * arky / 60)
        __m256 inhib = _mm256_mul_ps(_mm256_mul_ps(drive, arky), v_inv_60);
        __m256 suppressed = _mm256_max_ps(v_zero, _mm256_sub_ps(drive, inhib));
        _mm256_storeu_ps(striatal_out + i, suppressed);

        // Beta freq: 13.0 + da_def * 0.17
        __m256 f_res = _mm256_fmadd_ps(da_def, v_0_17, v_13);
        _mm256_storeu_ps(beta_freq + i, f_res);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> da_tone(N, 30.0f);
    std::vector<float> striatal_drive(N, 80.0f);
    std::vector<float> stop_active(N, 1.0f); // Stop signal active

    std::vector<float> arky_rate(N, 0.0f);
    std::vector<float> striatal_out(N, 0.0f);
    std::vector<float> beta_freq(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_gpe(da_tone.data(), striatal_drive.data(), stop_active.data(),
                           arky_rate.data(), striatal_out.data(), beta_freq.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_gpe(da_tone.data(), striatal_drive.data(), stop_active.data(),
                           arky_rate.data(), striatal_out.data(), beta_freq.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- GPe Subpopulations & Beta Resonance Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Arkypallidal Rate: " << arky_rate[0] << " Hz" << std::endl;
    std::cout << "Sample Striatal Drive (Post-Stop): " << striatal_out[0] << " %" << std::endl;
    std::cout << "Sample Beta Resonant Frequency: " << beta_freq[0] << " Hz" << std::endl;

    return 0;
}
