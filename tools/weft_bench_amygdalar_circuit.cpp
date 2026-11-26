// tools/weft_bench_amygdalar_circuit.cpp
// AVX2 SIMD Amygdalar BLA-CeA-ITC Conditioning & Extinction Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_amygdalar_avx2_bench(size_t num_units, size_t steps) {
    // 50% threat trials with low extinction (vmpfc = 0.1), 50% extinction trials (vmpfc = 0.9)
    std::vector<float> cs_drive(num_units, 0.85f);
    std::vector<float> w_bla(num_units, 0.70f);
    std::vector<float> out_cea(num_units, 0.0f);

    float us_shock = 1.0f; // Unconditioned teacher stimulus
    float vmpfc_extinction = 0.20f; // Lower default extinction to capture threat events
    float eta_ltp = 0.05f;
    float alpha_itc = 0.95f;
    float w_max = 1.0f;

    __m256 v_eta_ltp = _mm256_set1_ps(eta_ltp);
    __m256 v_us = _mm256_set1_ps(us_shock);
    __m256 v_itc_clamp = _mm256_set1_ps(alpha_itc * vmpfc_extinction);
    __m256 v_w_max = _mm256_set1_ps(w_max);
    __m256 v_zero = _mm256_setzero_ps();

    size_t threat_events = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_units; i += 8) {
            __m256 v_cs = _mm256_loadu_ps(&cs_drive[i]);
            __m256 v_w = _mm256_loadu_ps(&w_bla[i]);

            // LTP update: W_new = min(w_max, W + eta * CS * US)
            __m256 v_dw = _mm256_mul_ps(v_eta_ltp, _mm256_mul_ps(v_cs, v_us));
            __m256 v_w_new = _mm256_min_ps(v_w_max, _mm256_add_ps(v_w, v_dw));
            _mm256_storeu_ps(&w_bla[i], v_w_new);

            // BLA excitatory output: W_new * CS
            __m256 bla_out = _mm256_mul_ps(v_w_new, v_cs);

            // CeA threat output: max(0, BLA - ITC_clamp)
            __m256 cea_out = _mm256_max_ps(v_zero, _mm256_sub_ps(bla_out, v_itc_clamp));
            _mm256_storeu_ps(&out_cea[i], cea_out);

            __m256 threat_mask = _mm256_cmp_ps(cea_out, _mm256_set1_ps(0.2f), _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(threat_mask);
            threat_events += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_units) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Amygdalar Conditioning & Extinction Microbenchmark" << std::endl;
    std::cout << "  Amygdala Synaptic Units:  " << num_units << std::endl;
    std::cout << "  Timesteps:                " << steps << std::endl;
    std::cout << "  Elapsed Time:             " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:               " << std::fixed << std::setprecision(2) << millions_per_sec << " Million unit-steps/sec" << std::endl;
    std::cout << "  Threat Response Events:   " << threat_events << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t units = 65536; // 64k amygdalar circuits
    size_t steps = 1000;
    if (argc > 1) units = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_amygdalar_avx2_bench(units, steps);
    return 0;
}
