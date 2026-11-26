#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Pontomedullary Reticular Formation (PMRF) Reticulospinal Synergy Matrix
// Evaluates:
// 1. Bilateral axial & proximal muscle synergy vector recruitment:
//    M_act[m] = sum_k (W[m, k] * Synergy_drive[k])
// 2. Dynamic Center-of-Mass (CoM) stability margin under postural perturbation
// Target #376 in WEFT Verification Suite.

void benchmark_scalar_synergy(const std::vector<float>& pmrf_drive,
                              const std::vector<float>& com_disp,
                              const std::vector<float>& bos_margin,
                              std::vector<float>& axial_stiffness,
                              std::vector<float>& stability_idx,
                              size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float drive = pmrf_drive[i];
        axial_stiffness[i] = 30.0f + drive * 2.0f;
        float margin = bos_margin[i] - com_disp[i];
        stability_idx[i] = (margin > 0.0f) ? (margin / bos_margin[i]) : 0.0f;
    }
}

void benchmark_avx2_synergy(const float* pmrf_drive,
                            const float* com_disp,
                            const float* bos_margin,
                            float* axial_stiffness,
                            float* stability_idx,
                            size_t n) {
    const __m256 v_base_stiff = _mm256_set1_ps(30.0f);
    const __m256 v_slope = _mm256_set1_ps(2.0f);
    const __m256 v_zero = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 d = _mm256_loadu_ps(pmrf_drive + i);
        __m256 c = _mm256_loadu_ps(com_disp + i);
        __m256 b = _mm256_loadu_ps(bos_margin + i);

        // Axial stiffness: 30.0 + d * 2.0
        __m256 stiff = _mm256_fmadd_ps(d, v_slope, v_base_stiff);
        _mm256_storeu_ps(axial_stiffness + i, stiff);

        // Stability index: max(0.0, (b - c) / b)
        __m256 diff = _mm256_sub_ps(b, c);
        __m256 r_b = _mm256_rcp_ps(b);
        __m256 stab = _mm256_mul_ps(diff, r_b);
        stab = _mm256_max_ps(stab, v_zero);
        _mm256_storeu_ps(stability_idx + i, stab);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> pmrf_drive(N, 65.0f);
    std::vector<float> com_disp(N, 14.0f);
    std::vector<float> bos_margin(N, 50.0f);

    std::vector<float> axial_stiffness(N, 0.0f);
    std::vector<float> stability_idx(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_synergy(pmrf_drive.data(), com_disp.data(), bos_margin.data(),
                               axial_stiffness.data(), stability_idx.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_synergy(pmrf_drive.data(), com_disp.data(), bos_margin.data(),
                               axial_stiffness.data(), stability_idx.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Reticulospinal Postural Synergy Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Axial Stiffness: " << axial_stiffness[0] << " N*m/rad" << std::endl;
    std::cout << "Sample Stability Index: " << stability_idx[0] << std::endl;

    return 0;
}
