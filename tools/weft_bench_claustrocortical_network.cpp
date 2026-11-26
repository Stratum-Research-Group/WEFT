// tools/weft_bench_claustrocortical_network.cpp
// AVX2 SIMD Multi-Area Claustrocortical Salience Network Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_claustrocortical_avx2_bench(size_t num_columns, size_t steps) {
    std::vector<float> i_stim(num_columns, 15.0f);
    std::vector<float> s_cue(num_columns, 0.0f); // 20% salient columns, 80% distractors
    for (size_t i = 0; i < num_columns / 5; ++i) s_cue[i] = 1.0f;
    std::vector<float> i_eff(num_columns, 0.0f);

    float g_cl = 0.85f; // Strong claustrum salience gating
    float gamma_cl = 0.80f;
    float alpha_gaba = 0.85f;

    float gain_salient = 1.0f + gamma_cl * g_cl;
    float gain_distractor = std::max(0.05f, 1.0f - alpha_gaba * g_cl);

    __m256 v_gain_salient = _mm256_set1_ps(gain_salient);
    __m256 v_gain_distractor = _mm256_set1_ps(gain_distractor);
    __m256 v_half = _mm256_set1_ps(0.5f);

    size_t salient_bound_events = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_columns; i += 8) {
            __m256 v_s = _mm256_loadu_ps(&i_stim[i]);
            __m256 v_cue = _mm256_loadu_ps(&s_cue[i]);

            __m256 is_salient = _mm256_cmp_ps(v_cue, v_half, _CMP_GT_OQ);
            __m256 v_gain = _mm256_blendv_ps(v_gain_distractor, v_gain_salient, is_salient);

            __m256 v_out = _mm256_mul_ps(v_s, v_gain);
            _mm256_storeu_ps(&i_eff[i], v_out);

            __m256 bound_mask = _mm256_cmp_ps(v_out, _mm256_set1_ps(20.0f), _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(bound_mask);
            salient_bound_events += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_columns) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Claustrocortical Salience Microbenchmark" << std::endl;
    std::cout << "  Cortical Columns: " << num_columns << std::endl;
    std::cout << "  Timesteps:        " << steps << std::endl;
    std::cout << "  Elapsed Time:     " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:       " << std::fixed << std::setprecision(2) << millions_per_sec << " Million column-steps/sec" << std::endl;
    std::cout << "  Salient Bound Events: " << salient_bound_events << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t columns = 65536; // 64k cortical columns
    size_t steps = 1000;
    if (argc > 1) columns = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_claustrocortical_avx2_bench(columns, steps);
    return 0;
}
