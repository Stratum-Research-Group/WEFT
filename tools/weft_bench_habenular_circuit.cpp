// tools/weft_bench_habenular_circuit.cpp
// AVX2 SIMD Lateral Habenula - RMTg - Dopamine Dip Circuit Microbenchmark
// Stratum Research Group - Nicholas Charles Mynhardt

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include <cstdlib>

void run_habenular_avx2_bench(size_t num_units, size_t steps) {
    std::vector<float> rpe_vals(num_units, -0.60f); // Negative RPE (reward omission)
    std::vector<float> aversive_drive(num_units, 0.40f);
    std::vector<float> lhb_out(num_units, 0.0f);
    std::vector<float> da_out(num_units, 0.50f);

    float da_base = 0.50f;
    float beta_rpe = 1.50f;
    float gamma_rmtg = 0.40f;

    __m256 v_da_base = _mm256_set1_ps(da_base);
    __m256 v_beta_rpe = _mm256_set1_ps(beta_rpe);
    __m256 v_gamma_rmtg = _mm256_set1_ps(gamma_rmtg);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_ten = _mm256_set1_ps(10.0f);

    size_t da_dips = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t s = 0; s < steps; ++s) {
        for (size_t i = 0; i < num_units; i += 8) {
            __m256 v_rpe = _mm256_loadu_ps(&rpe_vals[i]);
            __m256 v_av = _mm256_loadu_ps(&aversive_drive[i]);

            // rpe_loss = max(0, -rpe * beta_rpe)
            __m256 neg_rpe = _mm256_sub_ps(v_zero, v_rpe);
            __m256 rpe_loss = _mm256_max_ps(v_zero, _mm256_mul_ps(neg_rpe, v_beta_rpe));

            // lhb_act = min(10.0, rpe_loss + 0.8 * aversive)
            __m256 lhb_act = _mm256_add_ps(rpe_loss, _mm256_mul_ps(v_av, _mm256_set1_ps(0.8f)));
            lhb_act = _mm256_min_ps(v_ten, lhb_act);
            _mm256_storeu_ps(&lhb_out[i], lhb_act);

            // da = clamp(da_base - gamma_rmtg * lhb_act, 0.0, 1.0)
            __m256 da = _mm256_sub_ps(v_da_base, _mm256_mul_ps(v_gamma_rmtg, lhb_act));
            da = _mm256_max_ps(v_zero, _mm256_min_ps(v_one, da));
            _mm256_storeu_ps(&da_out[i], da);

            // Detect dopamine pause (DA < 0.20)
            __m256 dip_mask = _mm256_cmp_ps(da, _mm256_set1_ps(0.20f), _CMP_LT_OQ);
            int mask_bits = _mm256_movemask_ps(dip_mask);
            da_dips += __builtin_popcount(mask_bits);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(t_end - t_start).count();
    double total_evaluations = static_cast<double>(num_units) * steps;
    double millions_per_sec = (total_evaluations / duration_sec) / 1e6;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "WEFT AVX2 Lateral Habenula - RMTg Microbenchmark" << std::endl;
    std::cout << "  Anti-Reward Units: " << num_units << std::endl;
    std::cout << "  Timesteps:         " << steps << std::endl;
    std::cout << "  Elapsed Time:      " << std::fixed << std::setprecision(4) << duration_sec << " s" << std::endl;
    std::cout << "  Throughput:        " << std::fixed << std::setprecision(2) << millions_per_sec << " Million unit-steps/sec" << std::endl;
    std::cout << "  Dopamine Dips:     " << da_dips << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    size_t units = 65536; // 64k LHb circuits
    size_t steps = 1000;
    if (argc > 1) units = std::stoull(argv[1]);
    if (argc > 2) steps = std::stoull(argv[2]);

    run_habenular_avx2_bench(units, steps);
    return 0;
}
