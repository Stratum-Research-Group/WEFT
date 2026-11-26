#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

// WEFT AVX2 Microbenchmark: AIP-PMv (F5) Visual Grasp Affordance & Preshaping Aperture
// Benchmarks 32,768 graspable visual 3D object instances.

void benchmark_grasp_affordance_avx2(size_t n, size_t iterations) {
    std::vector<float> object_width(n);
    std::vector<float> aip_act_out(n, 0.0f);
    std::vector<float> f5_drive_out(n, 0.0f);
    std::vector<float> peak_aperture_out(n, 0.0f);
    std::vector<float> mode_out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        object_width[i] = 5.0f + 70.0f * (static_cast<float>(i) / static_cast<float>(n));
    }

    const __m256 v_a = _mm256_set1_ps(18.0f);
    const __m256 v_b = _mm256_set1_ps(0.92f);
    const __m256 v_max_aperture = _mm256_set1_ps(125.0f);
    const __m256 v_pinch_thresh = _mm256_set1_ps(35.0f);
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_zero = _mm256_setzero_ps();

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < n; i += 8) {
            __m256 vw = _mm256_loadu_ps(&object_width[i]);

            // AIP activation: exp(-((w - 40)^2) / 1250)
            __m256 diff_aip = _mm256_sub_ps(vw, _mm256_set1_ps(40.0f));
            __m256 sq_diff = _mm256_mul_ps(diff_aip, diff_aip);
            __m256 x_aip = _mm256_div_ps(sq_diff, _mm256_set1_ps(1250.0f));
            // Rational exp(-x) approximation: 1 / (1 + x + 0.5 * x^2)
            __m256 denom_aip = _mm256_add_ps(v_one, _mm256_add_ps(x_aip, _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_mul_ps(x_aip, x_aip))));
            __m256 aip_act = _mm256_div_ps(v_one, denom_aip);

            // Peak aperture: a + b * w capped at 125.0
            __m256 raw_ap = _mm256_add_ps(v_a, _mm256_mul_ps(v_b, vw));
            __m256 peak_ap = _mm256_min_ps(raw_ap, v_max_aperture);

            // Grip mode: 1.0 if w >= 35.0 (Power), else 0.0 (Pinch)
            __m256 is_power = _mm256_cmp_ps(vw, v_pinch_thresh, _CMP_GE_OQ);
            __m256 mode_val = _mm256_blendv_ps(v_zero, v_one, is_power);

            // F5 preshaping drive: blend based on pinch vs power
            __m256 drive_pinch = _mm256_sub_ps(_mm256_set1_ps(0.95f), _mm256_mul_ps(_mm256_set1_ps(0.005f), vw));
            __m256 drive_power = _mm256_add_ps(_mm256_set1_ps(0.65f), _mm256_mul_ps(_mm256_set1_ps(0.003f), vw));
            __m256 f5_drive = _mm256_blendv_ps(drive_pinch, drive_power, is_power);

            _mm256_storeu_ps(&aip_act_out[i], aip_act);
            _mm256_storeu_ps(&f5_drive_out[i], f5_drive);
            _mm256_storeu_ps(&peak_aperture_out[i], peak_ap);
            _mm256_storeu_ps(&mode_out[i], mode_val);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "[WEFT AVX2 BENCH] Grasp Affordance Visual Instances: " << n << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Iterations: " << iterations << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Time per Iteration: " << (elapsed.count() / iterations) << " us" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Throughput: " 
              << (static_cast<double>(n * iterations) / (elapsed.count() * 1e-6)) / 1e6 << " MAffordanceInstances/sec" << std::endl;
    std::cout << "[WEFT AVX2 BENCH] Sample Target [n/2]: Width=" << object_width[n/2] << " mm => AIP=" 
              << aip_act_out[n/2] * 100.0f << "%, F5=" << f5_drive_out[n/2] * 100.0f << "%, PeakAperture=" 
              << peak_aperture_out[n/2] << " mm, Mode=" << (mode_out[n/2] > 0.5f ? "POWER" : "PINCH") << std::endl;
}

int main() {
    benchmark_grasp_affordance_avx2(32768, 5000);
    return 0;
}
