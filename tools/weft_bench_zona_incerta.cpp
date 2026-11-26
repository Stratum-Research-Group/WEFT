/**
 * tools/weft_bench_zona_incerta.cpp
 * AVX2/FMA Native Vectorized SIMD Microbenchmark
 * Zona Incerta (ZI) Visceromotor Integration & Binge Feeding Gating
 * Lean 4 Formal Target #394 | Paper #203 | Cycle v340.72.0
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <cmath>
#include <iomanip>

struct alignas(32) ZIBatch {
    float hunger[8];     // [0.0, 1.0]
    float prey[8];       // [0.0, 1.0]
    float stress[8];     // [0.0, 1.0]
};

void run_zi_avx2_fma(
    const float* __restrict__ hunger,
    const float* __restrict__ prey,
    const float* __restrict__ stress,
    float* __restrict__ out_pv_rate,
    float* __restrict__ out_som_rate,
    float* __restrict__ out_pag_disinh,
    float* __restrict__ out_binge_prob,
    size_t count
) {
    const __m256 c_pv_base   = _mm256_set1_ps(4.0f);
    const __m256 c_pv_hunger = _mm256_set1_ps(50.0f);
    const __m256 c_pv_stress = _mm256_set1_ps(15.0f);
    const __m256 c_pv_max    = _mm256_set1_ps(60.0f);
    const __m256 c_zero      = _mm256_setzero_ps();
    const __m256 c_one       = _mm256_set1_ps(1.0f);

    const __m256 c_som_base  = _mm256_set1_ps(3.0f);
    const __m256 c_som_scale = _mm256_set1_ps(45.0f);
    const __m256 c_som_max   = _mm256_set1_ps(50.0f);
    const __m256 c_half      = _mm256_set1_ps(0.5f);

    const __m256 c_pag_pv    = _mm256_set1_ps(0.6f / 50.0f);
    const __m256 c_pag_som   = _mm256_set1_ps(0.4f / 50.0f);

    const __m256 c_binge_slope = _mm256_set1_ps(0.12f);
    const __m256 c_binge_th    = _mm256_set1_ps(30.0f);

    for (size_t i = 0; i < count; i += 8) {
        __m256 v_hunger = _mm256_loadu_ps(hunger + i);
        __m256 v_prey   = _mm256_loadu_ps(prey + i);
        __m256 v_stress = _mm256_loadu_ps(stress + i);

        // 1. Rostral PV+ rate: pv = clamp(4.0 + 50.0*hunger - 15.0*stress, 0.0, 60.0)
        __m256 v_pv = _mm256_fmadd_ps(c_pv_hunger, v_hunger, c_pv_base);
        v_pv = _mm256_fnmadd_ps(c_pv_stress, v_stress, v_pv);
        v_pv = _mm256_max_ps(c_zero, _mm256_min_ps(c_pv_max, v_pv));

        // 2. Caudal SOM+ rate: som = clamp(3.0 + 45.0*prey*(0.5 + 0.5*hunger), 0.0, 50.0)
        __m256 v_amp = _mm256_fmadd_ps(c_half, v_hunger, c_half);
        __m256 v_prey_amp = _mm256_mul_ps(v_prey, v_amp);
        __m256 v_som = _mm256_fmadd_ps(c_som_scale, v_prey_amp, c_som_base);
        v_som = _mm256_max_ps(c_zero, _mm256_min_ps(c_som_max, v_som));

        // 3. PAG Disinhibition: (0.6*pv + 0.4*som)/50.0 clamped to [0, 1]
        __m256 v_pag = _mm256_mul_ps(v_pv, c_pag_pv);
        v_pag = _mm256_fmadd_ps(v_som, c_pag_som, v_pag);
        v_pag = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_pag));

        // 4. Binge feeding probability: clamp(0.5 + 0.12*(pv - 30.0), 0.0, 1.0)
        __m256 v_diff = _mm256_sub_ps(v_pv, c_binge_th);
        __m256 v_binge = _mm256_fmadd_ps(c_binge_slope, v_diff, c_half);
        v_binge = _mm256_max_ps(c_zero, _mm256_min_ps(c_one, v_binge));

        _mm256_storeu_ps(out_pv_rate + i, v_pv);
        _mm256_storeu_ps(out_som_rate + i, v_som);
        _mm256_storeu_ps(out_pag_disinh + i, v_pag);
        _mm256_storeu_ps(out_binge_prob + i, v_binge);
    }
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "WEFT AVX2/FMA SIMD Microbenchmark: Zona Incerta Visceromotor Gate" << std::endl;
    std::cout << "Lean 4 Target #394 | Paper #203 | Cycle v340.72.0" << std::endl;
    std::cout << "================================================================================" << std::endl;

    const size_t N = 1000000; // 1,000,000 iterations
    std::vector<float> hunger(N, 0.75f);
    std::vector<float> prey(N, 0.40f);
    std::vector<float> stress(N, 0.15f);

    std::vector<float> out_pv(N);
    std::vector<float> out_som(N);
    std::vector<float> out_pag(N);
    std::vector<float> out_binge(N);

    // Warmup
    for (int w = 0; w < 5; ++w) {
        run_zi_avx2_fma(hunger.data(), prey.data(), stress.data(),
                        out_pv.data(), out_som.data(), out_pag.data(), out_binge.data(), N);
    }

    // Benchmark
    const int RUNS = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        run_zi_avx2_fma(hunger.data(), prey.data(), stress.data(),
                        out_pv.data(), out_som.data(), out_pag.data(), out_binge.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / RUNS;
    double throughput = (double)N / (avg_ms * 1e-3) / 1e6; // Melem/s

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Processed Elements:  " << N << " (8-lane AVX2/FMA SIMD)" << std::endl;
    std::cout << "Average Latency:     " << avg_ms << " ms" << std::endl;
    std::cout << "Peak Throughput:     " << throughput << " Melem/s (" << (throughput / 1e3) << " Gelem/s)" << std::endl;
    std::cout << "\nSample Verification at index 0:" << std::endl;
    std::cout << "  Hunger:             " << hunger[0] * 100.0f << "%" << std::endl;
    std::cout << "  Prey Proximity:     " << prey[0] * 100.0f << "%" << std::endl;
    std::cout << "  Stress Tone:        " << stress[0] * 100.0f << "%" << std::endl;
    std::cout << "  Rostral PV+ Rate:   " << out_pv[0] << " Hz" << std::endl;
    std::cout << "  Caudal SOM+ Rate:   " << out_som[0] << " Hz" << std::endl;
    std::cout << "  PAG Disinhibition:  " << out_pag[0] * 100.0f << "%" << std::endl;
    std::cout << "  Binge Probability:  " << out_binge[0] * 100.0f << "%" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
