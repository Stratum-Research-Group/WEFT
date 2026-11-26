#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Pedunculopontine Nucleus (PPN) & Spinal Locomotor CPG Engine
// Evaluates:
// 1. Net locomotor drive: net = max(0, glut_drive - bg_clamp * 1.1)
// 2. Stepping cadence: cadence = (net >= 15.0) ? (0.8 + (net - 15.0) * 0.0235) : 0.0
// 3. Forward locomotor speed: speed = cadence * (0.6 + cadence * 0.285)
// Target #384 in WEFT Verification Suite.

void benchmark_scalar_ppn(const std::vector<float>& glut_drive,
                          const std::vector<float>& bg_clamp,
                          std::vector<float>& net_drive,
                          std::vector<float>& cadence_hz,
                          std::vector<float>& speed_mps,
                          size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float g = glut_drive[i];
        float b = bg_clamp[i];
        float net = std::fmax(0.0f, g - b * 1.1f);
        net_drive[i] = net;

        float cadence = (net >= 15.0f) ? std::fmin(2.8f, 0.8f + (net - 15.0f) * 0.023529f) : 0.0f;
        cadence_hz[i] = cadence;

        float stride = 0.6f + (cadence / 2.8f) * 0.8f;
        speed_mps[i] = stride * cadence;
    }
}

void benchmark_avx2_ppn(const float* glut_drive,
                        const float* bg_clamp,
                        float* net_drive,
                        float* cadence_hz,
                        float* speed_mps,
                        size_t n) {
    const __m256 v_1_1 = _mm256_set1_ps(1.1f);
    const __m256 v_15 = _mm256_set1_ps(15.0f);
    const __m256 v_0_8 = _mm256_set1_ps(0.8f);
    const __m256 v_slope = _mm256_set1_ps(0.023529f);
    const __m256 v_2_8 = _mm256_set1_ps(2.8f);
    const __m256 v_0_6 = _mm256_set1_ps(0.6f);
    const __m256 v_stride_slope = _mm256_set1_ps(0.8f / 2.8f);
    const __m256 v_zero = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8) {
        __m256 g = _mm256_loadu_ps(glut_drive + i);
        __m256 b = _mm256_loadu_ps(bg_clamp + i);

        // Net: max(0, g - b * 1.1)
        __m256 net = _mm256_max_ps(v_zero, _mm256_sub_ps(g, _mm256_mul_ps(b, v_1_1)));
        _mm256_storeu_ps(net_drive + i, net);

        // Cadence: min(2.8, 0.8 + (net - 15.0) * slope)
        __m256 active_mask = _mm256_cmp_ps(net, v_15, _CMP_GE_OQ);
        __m256 cad_raw = _mm256_fmadd_ps(_mm256_sub_ps(net, v_15), v_slope, v_0_8);
        __m256 cad_clamped = _mm256_min_ps(v_2_8, cad_raw);
        __m256 cadence = _mm256_blendv_ps(v_zero, cad_clamped, active_mask);
        _mm256_storeu_ps(cadence_hz + i, cadence);

        // Speed: (0.6 + cadence * (0.8 / 2.8)) * cadence
        __m256 stride = _mm256_fmadd_ps(cadence, v_stride_slope, v_0_6);
        __m256 speed = _mm256_mul_ps(stride, cadence);
        _mm256_storeu_ps(speed_mps + i, speed);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> glut_drive(N, 65.0f);
    std::vector<float> bg_clamp(N, 20.0f);
    std::vector<float> net_drive(N, 0.0f);
    std::vector<float> cadence_hz(N, 0.0f);
    std::vector<float> speed_mps(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_ppn(glut_drive.data(), bg_clamp.data(),
                           net_drive.data(), cadence_hz.data(), speed_mps.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_ppn(glut_drive.data(), bg_clamp.data(),
                           net_drive.data(), cadence_hz.data(), speed_mps.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Pedunculopontine Nucleus (PPN) Locomotor CPG Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Net Locomotor Drive: " << net_drive[0] << " %" << std::endl;
    std::cout << "Sample Stepping Cadence: " << cadence_hz[0] << " Hz" << std::endl;
    std::cout << "Sample Forward Speed: " << speed_mps[0] << " m/s (" << speed_mps[0] * 3.6f << " km/h)" << std::endl;

    return 0;
}
