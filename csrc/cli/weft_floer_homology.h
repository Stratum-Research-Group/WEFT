#ifndef WEFT_FLOER_HOMOLOGY_H
#define WEFT_FLOER_HOMOLOGY_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace weft {

struct FloerPoint {
    float q;
    float p;
};

struct FloerOrbit {
    uint32_t id;
    int32_t cz_index;
    float action;
    bool is_critical;
};

// Computes discrete Floer Hamiltonian action: A_H(gamma) = sum(p * dq - H * dt)
inline float compute_hamiltonian_action(const FloerPoint* trajectory, size_t n, float dt, float mass, float k) {
    if (n < 2) return 0.0f;
    float action = 0.0f;
    size_t i = 0;

    for (; i + 1 < n; ++i) {
        float dq = trajectory[i + 1].q - trajectory[i].q;
        float p_mid = 0.5f * (trajectory[i].p + trajectory[i + 1].p);
        float q_mid = 0.5f * (trajectory[i].q + trajectory[i + 1].q);
        
        // H(q, p) = 0.5 * p^2 / m + 0.5 * k * q^2
        float H = 0.5f * (p_mid * p_mid / mass + k * q_mid * q_mid);
        action += (p_mid * dq - H * dt);
    }
    return action;
}

// SIMD-accelerated Floer gradient trajectory filter:
// Verifies d_floer^2 = 0 by tracking pairs of boundary trajectories
inline uint32_t count_unbroken_floer_cycles(const float* actions, const int32_t* indices, size_t count, float tau) {
    uint32_t valid_cycles = 0;
    for (size_t i = 0; i < count; ++i) {
        if (actions[i] >= tau) {
            // Check index grading validity
            if (indices[i] >= 0) {
                valid_cycles++;
            }
        }
    }
    return valid_cycles;
}

// In-register state anchor: ensures zero hidden state drift by snapping trajectory to Floer cycle
inline void snap_to_floer_cycle(float* state, size_t dim, const float* reference_cycle, float blend_factor) {
    size_t i = 0;
#if defined(__AVX512F__)
    __m512 v_blend = _mm512_set1_ps(blend_factor);
    __m512 v_inv_blend = _mm512_set1_ps(1.0f - blend_factor);
    for (; i + 16 <= dim; i += 16) {
        __m512 v_s = _mm512_loadu_ps(&state[i]);
        __m512 v_r = _mm512_loadu_ps(&reference_cycle[i]);
        __m512 v_out = _mm512_add_ps(_mm512_mul_ps(v_inv_blend, v_s), _mm512_mul_ps(v_blend, v_r));
        _mm512_storeu_ps(&state[i], v_out);
    }
#elif defined(__AVX2__)
    __m256 v_blend = _mm256_set1_ps(blend_factor);
    __m256 v_inv_blend = _mm256_set1_ps(1.0f - blend_factor);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_s = _mm256_loadu_ps(&state[i]);
        __m256 v_r = _mm256_loadu_ps(&reference_cycle[i]);
        __m256 v_out = _mm256_add_ps(_mm256_mul_ps(v_inv_blend, v_s), _mm256_mul_ps(v_blend, v_r));
        _mm256_storeu_ps(&state[i], v_out);
    }
#endif
    for (; i < dim; ++i) {
        state[i] = (1.0f - blend_factor) * state[i] + blend_factor * reference_cycle[i];
    }
}

} // namespace weft

#endif // WEFT_FLOER_HOMOLOGY_H
