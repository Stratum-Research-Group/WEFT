// weft_hamiltonian_ssm.h
// Native AVX-512 Symplectic Leapfrog Integrator for Continuous-Time Hamiltonian State Routing.
// Author: Nicholas Charles Mynhardt, Stratum Research Group (research@stratum.org)
// License: Apache-2.0

#pragma once

#include <immintrin.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>

namespace weft {

struct HamiltonianState {
    int dim;
    std::vector<float> q; // Position state (hidden activation)
    std::vector<float> p; // Conjugate momentum state

    explicit HamiltonianState(int d) : dim(d), q(d, 0.0f), p(d, 0.0f) {}
};

// Compute quadratic harmonic kinetic and potential energy: H = 0.5 * sum(p^2 + q^2)
inline float compute_hamiltonian_energy(const HamiltonianState& state) {
    const int d = state.dim;
    __m512 h_acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 vq = _mm512_loadu_ps(&state.q[i]);
        __m512 vp = _mm512_loadu_ps(&state.p[i]);
        h_acc = _mm512_fmadd_ps(vq, vq, h_acc);
        h_acc = _mm512_fmadd_ps(vp, vp, h_acc);
    }
    float energy = 0.5f * _mm512_reduce_add_ps(h_acc);
    for (; i < d; ++i) {
        energy += 0.5f * (state.q[i] * state.q[i] + state.p[i] * state.p[i]);
    }
    return energy;
}

// AVX-512 Fused Symplectic Störmer-Verlet Leapfrog Step:
// Potential force grad_V(q) = omega^2 * q (harmonic baseline for state-space stability)
inline void leapfrog_step_avx512(HamiltonianState& state, float dt, float omega_sq = 1.0f) {
    const int d = state.dim;
    const float half_dt = 0.5f * dt;
    __m512 v_half_dt = _mm512_set1_ps(half_dt);
    __m512 v_dt = _mm512_set1_ps(dt);
    __m512 v_omega_sq = _mm512_set1_ps(omega_sq);

    int i = 0;
    for (; i + 16 <= d; i += 16) {
        __m512 vq = _mm512_loadu_ps(&state.q[i]);
        __m512 vp = _mm512_loadu_ps(&state.p[i]);

        // 1. Half step momentum: p_{l+1/2} = p_l - half_dt * (omega_sq * q_l)
        __m512 force1 = _mm512_mul_ps(v_omega_sq, vq);
        vp = _mm512_fnmadd_ps(v_half_dt, force1, vp);

        // 2. Full step position: q_{l+1} = q_l + dt * p_{l+1/2}
        vq = _mm512_fmadd_ps(v_dt, vp, vq);

        // 3. Half step momentum: p_{l+1} = p_{l+1/2} - half_dt * (omega_sq * q_{l+1})
        __m512 force2 = _mm512_mul_ps(v_omega_sq, vq);
        vp = _mm512_fnmadd_ps(v_half_dt, force2, vp);

        _mm512_storeu_ps(&state.q[i], vq);
        _mm512_storeu_ps(&state.p[i], vp);
    }

    for (; i < d; ++i) {
        float f1 = omega_sq * state.q[i];
        state.p[i] -= half_dt * f1;
        state.q[i] += dt * state.p[i];
        float f2 = omega_sq * state.q[i];
        state.p[i] -= half_dt * f2;
    }
}

} // namespace weft
