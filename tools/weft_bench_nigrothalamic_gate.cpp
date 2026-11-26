#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Substantia Nigra Pars Reticulata (SNr) & Motor Thalamus Disinhibition Engine
// Evaluates:
// 1. Striatonigral dSPN pause-mediated SNr suppression: snr = max(0, tonic - dspn * tonic)
// 2. Thalamic membrane potential depolarization & T-type Ca2+ channel de-inactivation
// 3. Post-inhibitory rebound burst frequency: burst = (snr < 15.0) ? 250.0 * dspn : 0.0
// Target #382 in WEFT Verification Suite.

void benchmark_scalar_nigrothalamic(const std::vector<float>& dspn,
                                    const std::vector<float>& tonic_snr,
                                    std::vector<float>& snr_out,
                                    std::vector<float>& thalamic_vm,
                                    std::vector<float>& rebound_burst,
                                    size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float d = dspn[i];
        float t = tonic_snr[i];
        float s = std::fmax(0.0f, t - d * t * 1.2f);
        snr_out[i] = s;
        thalamic_vm[i] = -75.0f + ((t - s) / t) * 25.0f;
        rebound_burst[i] = (s < 15.0f) ? 250.0f * d : 0.0f;
    }
}

void benchmark_avx2_nigrothalamic(const float* dspn,
                                  const float* tonic_snr,
                                  float* snr_out,
                                  float* thalamic_vm,
                                  float* rebound_burst,
                                  size_t n) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_1_2 = _mm256_set1_ps(1.2f);
    const __m256 v_25 = _mm256_set1_ps(25.0f);
    const __m256 v_neg_75 = _mm256_set1_ps(-75.0f);
    const __m256 v_15 = _mm256_set1_ps(15.0f);
    const __m256 v_250 = _mm256_set1_ps(250.0f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 d = _mm256_loadu_ps(dspn + i);
        __m256 t = _mm256_loadu_ps(tonic_snr + i);

        // Inhibition: d * t * 1.2
        __m256 inhib = _mm256_mul_ps(_mm256_mul_ps(d, t), v_1_2);
        __m256 s = _mm256_max_ps(v_zero, _mm256_sub_ps(t, inhib));
        _mm256_storeu_ps(snr_out + i, s);

        // Thalamic Vm: -75.0 + ((t - s) / t) * 25.0
        __m256 diff = _mm256_sub_ps(t, s);
        __m256 r_t = _mm256_rcp_ps(t);
        __m256 vm = _mm256_fmadd_ps(_mm256_mul_ps(diff, r_t), v_25, v_neg_75);
        _mm256_storeu_ps(thalamic_vm + i, vm);

        // Rebound burst: (s < 15.0) ? 250.0 * d : 0.0
        __m256 mask = _mm256_cmp_ps(s, v_15, _CMP_LT_OQ);
        __m256 burst = _mm256_blendv_ps(v_zero, _mm256_mul_ps(v_250, d), mask);
        _mm256_storeu_ps(rebound_burst + i, burst);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> dspn(N, 0.70f);
    std::vector<float> tonic_snr(N, 65.0f);
    std::vector<float> snr_out(N, 0.0f);
    std::vector<float> thalamic_vm(N, 0.0f);
    std::vector<float> rebound_burst(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_nigrothalamic(dspn.data(), tonic_snr.data(),
                                     snr_out.data(), thalamic_vm.data(), rebound_burst.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_nigrothalamic(dspn.data(), tonic_snr.data(),
                                     snr_out.data(), thalamic_vm.data(), rebound_burst.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Substantia Nigra Disinhibition & Thalamic Rebound Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample SNr Output: " << snr_out[0] << " Hz" << std::endl;
    std::cout << "Sample Thalamic Vm: " << thalamic_vm[0] << " mV" << std::endl;
    std::cout << "Sample Thalamic Rebound Burst: " << rebound_burst[0] << " Hz" << std::endl;

    return 0;
}
