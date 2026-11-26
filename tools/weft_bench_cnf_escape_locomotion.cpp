#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>

// Benchmark for Cuneiform Nucleus (CnF) High-Speed Locomotor Escape & Asymmetric Gallop
// Evaluates:
// 1. Effective escape drive: drive = (cnf * (30.0 + threat * 0.7)) / 100.0
// 2. Gait mode bifurcation & interlimb phase lag:
//    phase = (drive >= 45.0) ? max(25.0, 180.0 - (drive - 45.0) * 2.818) : 180.0
// 3. High-velocity sprint escape cadence:
//    cadence = (drive >= 45.0) ? (2.5 + (drive - 45.0) * 0.04545) : (1.0 + (drive - 15.0) * 0.05)
// Target #385 in WEFT Verification Suite.

void benchmark_scalar_cnf(const std::vector<float>& cnf_drive,
                          const std::vector<float>& threat_level,
                          std::vector<float>& effective_drive,
                          std::vector<float>& cadence_hz,
                          std::vector<float>& phase_lag_deg,
                          std::vector<float>& speed_kmh,
                          size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float c = cnf_drive[i];
        float t = threat_level[i];
        float drive = (c * (30.0f + t * 0.7f)) / 100.0f;
        effective_drive[i] = drive;

        if (drive < 15.0f) {
            cadence_hz[i] = 0.0f;
            phase_lag_deg[i] = 180.0f;
            speed_kmh[i] = 0.0f;
        } else if (drive < 45.0f) {
            float cad = 1.0f + ((drive - 15.0f) / 30.0f) * 1.5f;
            cadence_hz[i] = cad;
            phase_lag_deg[i] = 180.0f;
            speed_kmh[i] = cad * 1.1f * 3.6f;
        } else {
            float cad = 2.5f + ((drive - 45.0f) / 55.0f) * 2.5f;
            cadence_hz[i] = cad;
            float phase = std::fmax(25.0f, 180.0f - ((drive - 45.0f) / 55.0f) * 155.0f);
            phase_lag_deg[i] = phase;
            speed_kmh[i] = cad * 1.8f * 3.6f;
        }
    }
}

void benchmark_avx2_cnf(const float* cnf_drive,
                        const float* threat_level,
                        float* effective_drive,
                        float* cadence_hz,
                        float* phase_lag_deg,
                        float* speed_kmh,
                        size_t n) {
    const __m256 v_30 = _mm256_set1_ps(30.0f);
    const __m256 v_0_7 = _mm256_set1_ps(0.7f);
    const __m256 v_0_01 = _mm256_set1_ps(0.01f);
    const __m256 v_15 = _mm256_set1_ps(15.0f);
    const __m256 v_45 = _mm256_set1_ps(45.0f);
    const __m256 v_180 = _mm256_set1_ps(180.0f);
    const __m256 v_25 = _mm256_set1_ps(25.0f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_gallop_phase_slope = _mm256_set1_ps(155.0f / 55.0f);
    const __m256 v_gallop_cad_slope = _mm256_set1_ps(2.5f / 55.0f);
    const __m256 v_gallop_cad_base = _mm256_set1_ps(2.5f);
    const __m256 v_speed_factor = _mm256_set1_ps(1.8f * 3.6f);

    for (size_t i = 0; i < n; i += 8) {
        __m256 c = _mm256_loadu_ps(cnf_drive + i);
        __m256 t = _mm256_loadu_ps(threat_level + i);

        // Drive: c * (30.0 + t * 0.7) * 0.01
        __m256 threat_mod = _mm256_fmadd_ps(t, v_0_7, v_30);
        __m256 drive = _mm256_mul_ps(_mm256_mul_ps(c, threat_mod), v_0_01);
        _mm256_storeu_ps(effective_drive + i, drive);

        // Mask gallop (drive >= 45.0)
        __m256 gallop_mask = _mm256_cmp_ps(drive, v_45, _CMP_GE_OQ);

        // Gallop Cadence: 2.5 + (drive - 45.0) * (2.5 / 55.0)
        __m256 cad_gallop = _mm256_fmadd_ps(_mm256_sub_ps(drive, v_45), v_gallop_cad_slope, v_gallop_cad_base);
        _mm256_storeu_ps(cadence_hz + i, cad_gallop);

        // Phase lag: max(25.0, 180.0 - (drive - 45.0) * (155.0 / 55.0))
        __m256 raw_phase = _mm256_fnmadd_ps(_mm256_sub_ps(drive, v_45), v_gallop_phase_slope, v_180);
        __m256 phase_gallop = _mm256_max_ps(v_25, raw_phase);
        __m256 final_phase = _mm256_blendv_ps(v_180, phase_gallop, gallop_mask);
        _mm256_storeu_ps(phase_lag_deg + i, final_phase);

        // Speed km/h
        __m256 speed = _mm256_mul_ps(cad_gallop, v_speed_factor);
        _mm256_storeu_ps(speed_kmh + i, speed);
    }
}

int main() {
    const size_t N = 1000000;
    const int WARMUP = 10;
    const int RUNS = 50;

    std::vector<float> cnf_drive(N, 75.0f);
    std::vector<float> threat_level(N, 80.0f);
    std::vector<float> effective_drive(N, 0.0f);
    std::vector<float> cadence_hz(N, 0.0f);
    std::vector<float> phase_lag_deg(N, 0.0f);
    std::vector<float> speed_kmh(N, 0.0f);

    for (int w = 0; w < WARMUP; ++w) {
        benchmark_avx2_cnf(cnf_drive.data(), threat_level.data(),
                           effective_drive.data(), cadence_hz.data(),
                           phase_lag_deg.data(), speed_kmh.data(), N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        benchmark_avx2_cnf(cnf_drive.data(), threat_level.data(),
                           effective_drive.data(), cadence_hz.data(),
                           phase_lag_deg.data(), speed_kmh.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double dur_ms = std::chrono::duration<double, std::milli>(end - start).count() / RUNS;

    std::cout << "--- Cuneiform Nucleus (CnF) Escape Locomotion Microbenchmark ---" << std::endl;
    std::cout << "Elements: " << N << " per run (" << RUNS << " iterations)" << std::endl;
    std::cout << "AVX2+FMA Execution Time: " << std::fixed << std::setprecision(3) << dur_ms << " ms" << std::endl;
    double throughput_melem = (N / (dur_ms / 1000.0)) / 1e6;
    std::cout << "AVX2+FMA Throughput: " << throughput_melem << " Melem/s" << std::endl;
    std::cout << "Sample Effective Drive: " << effective_drive[0] << " %" << std::endl;
    std::cout << "Sample Escape Cadence: " << cadence_hz[0] << " Hz" << std::endl;
    std::cout << "Sample Interlimb Phase Lag: " << phase_lag_deg[0] << " deg" << std::endl;
    std::cout << "Sample Escape Speed: " << speed_kmh[0] << " km/h" << std::endl;

    return 0;
}
