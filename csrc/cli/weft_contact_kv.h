// weft_contact_kv.h
// Native AVX-512 in-register Contact Hamiltonian Leapfrog Integrator for Dissipation-Bounded KV Dynamics.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace weft {

// Contact state in R^{2d + 1}: position q (dim d), momentum p (dim d), dissipation coordinate z (scalar)
struct ContactState {
    int d;
    std::vector<float> q;
    std::vector<float> p;
    float z;

    explicit ContactState(int dim) : d(dim), q(dim, 0.0f), p(dim, 0.0f), z(0.0f) {}

    float kinetic_energy() const {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) sum += 0.5f * p[i] * p[i];
        return sum;
    }

    float potential_energy() const {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) sum += 0.5f * q[i] * q[i];
        return sum;
    }

    float total_energy() const {
        return kinetic_energy() + potential_energy();
    }
};

// Contact leapfrog step with damping parameter gamma >= 0:
// q_{t+1} = q_t + dt * p_{t+1/2}
// p_{t+1} = (1 - gamma * dt) * p_t - dt * q_t
// z_{t+1} = z_t + dt * (kinetic - potential - gamma * z_t)
inline void contact_leapfrog_step_avx512(ContactState& state, float dt, float gamma = 0.05f) {
    const int d = state.d;
    const float damp_factor = 1.0f - gamma * dt;

    __m512 v_dt = _mm512_set1_ps(dt);
    __m512 v_damp = _mm512_set1_ps(damp_factor);

    float p_sq_sum = 0.0f;
    float q_sq_sum = 0.0f;

    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 vq = _mm512_loadu_ps(&state.q[i]);
        __m512 vp = _mm512_loadu_ps(&state.p[i]);

        // Damped momentum update: p_new = damp_factor * p - dt * q
        __m512 vp_damped = _mm512_mul_ps(v_damp, vp);
        __m512 vp_new = _mm512_fnmadd_ps(v_dt, vq, vp_damped);

        // Position update: q_new = q + dt * vp_new
        __m512 vq_new = _mm512_fmadd_ps(v_dt, vp_new, vq);

        _mm512_storeu_ps(&state.q[i], vq_new);
        _mm512_storeu_ps(&state.p[i], vp_new);
    }

    for (; i < d; ++i) {
        state.p[i] = damp_factor * state.p[i] - dt * state.q[i];
        state.q[i] += dt * state.p[i];
        p_sq_sum += 0.5f * state.p[i] * state.p[i];
        q_sq_sum += 0.5f * state.q[i] * state.q[i];
    }

    // Auxiliary contact coordinate z tracks cumulative dissipation
    float kinetic = state.kinetic_energy();
    float potential = state.potential_energy();
    state.z += dt * (kinetic - potential - gamma * state.z);
}

} // namespace weft
